from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"

MODEL_DIRS = {
    "binary": PROJECT_ROOT / "artifacts" / "final_binary",
    "multiclass_best": PROJECT_ROOT / "artifacts" / "final_multiclass",
    "multiclass_nn": PROJECT_ROOT / "artifacts" / "final_multiclass_nn",
    "class33": PROJECT_ROOT / "artifacts" / "final_33class",
}


def find_existing_path(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


DATASET_FILES = {
    "binary": find_existing_path([
        PROJECT_ROOT / "clean_sample.csv",
        PROJECT_ROOT / "dataset" / "clean_sample.csv",
        PROJECT_ROOT / "dataset" / "binary_v1" / "clean_sample.csv",
        PROJECT_ROOT / "data" / "clean_sample.csv",
    ]),
    "multiclass": find_existing_path([
        PROJECT_ROOT / "dataset" / "multiclass_v1" / "test.csv",
    ]),
    "class33": find_existing_path([
        PROJECT_ROOT / "dataset" / "multiclass_33_v1" / "test.csv",
    ]),
}


def to_python(value: Any):
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]

    return value


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("Final_Label", "Label"):
        if c in df.columns:
            return c
    return None


def map_prediction_to_label(pred_value: Any, label_mapping: dict) -> str:
    if isinstance(pred_value, str):
        return pred_value
    idx_to_label = {int(v): k for k, v in label_mapping.items()}
    return idx_to_label.get(int(pred_value), str(pred_value))


def extract_metric(metrics: dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except Exception:
                pass

    nested_candidates = [
        metrics.get("validation"),
        metrics.get("test"),
        metrics.get("val"),
        metrics.get("results"),
    ]

    for nested in nested_candidates:
        if isinstance(nested, dict):
            for k in keys:
                if k in nested and nested[k] is not None:
                    try:
                        return float(nested[k])
                    except Exception:
                        pass

    return None


def latest_run_metrics(prefix: str) -> Optional[tuple[Path, dict]]:
    candidates = []
    if not RUNS_ROOT.exists():
        return None

    for path in RUNS_ROOT.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            metrics_path = path / "metrics.json"
            if metrics_path.exists():
                candidates.append((path, metrics_path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0].name, reverse=True)

    for run_dir, metrics_path in candidates:
        try:
            return run_dir, load_json(metrics_path)
        except Exception:
            continue

    return None


def best_tuned_nn_metrics() -> Optional[dict]:
    candidates = []
    if not RUNS_ROOT.exists():
        return None

    for run_dir in RUNS_ROOT.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("mlp_tuned_multiclass_v1_"):
            metrics_path = run_dir / "promotion_ready_metrics.json"
            if not metrics_path.exists():
                metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue

            try:
                metrics = load_json(metrics_path)
                score = extract_metric(metrics, ["val_macro_f1", "test_macro_f1", "macro_f1"])
                if score is None:
                    continue
                candidates.append((score, run_dir, metrics_path, metrics))
            except Exception:
                continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, run_dir, metrics_path, metrics = candidates[0]

    return {
        "run_dir": run_dir,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "score": score,
    }


class ModelBundle:
    def __init__(self, model_dir: Path):
        self.model = joblib.load(model_dir / "best_model.joblib")
        self.feature_order = load_json(model_dir / "feature_order.json")
        self.label_mapping = load_json(model_dir / "label_mapping.json")

    def predict_from_row(self, row: pd.Series):
        missing = [f for f in self.feature_order if f not in row.index]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        features = {f: to_python(row[f]) for f in self.feature_order}
        X = pd.DataFrame([features], columns=self.feature_order)

        pred = self.model.predict(X)[0]
        label = map_prediction_to_label(pred, self.label_mapping)

        confidence = None
        probabilities = None

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            confidence = float(max(probs))
            idx_to_label = {int(v): k for k, v in self.label_mapping.items()}
            probabilities = {
                idx_to_label.get(i, str(i)): float(p) for i, p in enumerate(probs)
            }

        return {
            "features": features,
            "prediction_label": to_python(label),
            "confidence": to_python(confidence),
            "probabilities": to_python(probabilities),
        }


class DatasetStepper:
    def __init__(self, path: Path):
        self.df = pd.read_csv(path)
        self.index = 0
        self.label_col = detect_label_column(self.df)

    def next_row(self):
        row = self.df.iloc[self.index % len(self.df)]
        row_index = self.index % len(self.df)
        self.index += 1

        actual = None
        if self.label_col:
            actual = row[self.label_col]

        return {
            "row_index": int(row_index),
            "row": row,
            "actual": to_python(actual),
            "raw_record": to_python(row.to_dict()),
        }


class Runtime:
    def __init__(self):
        self.models = {}
        self.datasets = {}
        self.logs = {
            "binary": deque(maxlen=100),
            "multiclass_best": deque(maxlen=100),
            "multiclass_nn": deque(maxlen=100),
            "class33": deque(maxlen=100),
        }

    def load_all(self):
        errors = []

        for key, path in MODEL_DIRS.items():
            try:
                self.models[key] = ModelBundle(path)
            except Exception as e:
                errors.append(f"{key}: {e}")

        for key, path in DATASET_FILES.items():
            if path:
                try:
                    self.datasets[key] = DatasetStepper(path)
                except Exception as e:
                    errors.append(f"{key}: {e}")
            else:
                errors.append(f"{key}: dataset not found")

        return errors

    def build_entry(self, actual, prediction_packet, row_index, raw_record):
        predicted = prediction_packet["prediction_label"]
        return {
            "row_index": int(row_index),
            "actual": to_python(actual),
            "predicted": to_python(predicted),
            "correct": None if actual is None else str(actual) == str(predicted),
            "confidence": to_python(prediction_packet["confidence"]),
            "features": to_python(prediction_packet["features"]),
            "raw_record": to_python(raw_record),
            "probabilities": to_python(prediction_packet["probabilities"]),
        }

    def tick(self):
        results = {}
        errors = {}

        try:
            if "binary" in self.models and "binary" in self.datasets:
                packet = self.datasets["binary"].next_row()
                pred = self.models["binary"].predict_from_row(packet["row"])
                entry = self.build_entry(packet["actual"], pred, packet["row_index"], packet["raw_record"])
                self.logs["binary"].appendleft(entry)
                results["binary"] = entry
        except Exception as e:
            errors["binary"] = str(e)

        try:
            if "multiclass" in self.datasets:
                packet = self.datasets["multiclass"].next_row()

                if "multiclass_best" in self.models:
                    pred = self.models["multiclass_best"].predict_from_row(packet["row"])
                    entry = self.build_entry(packet["actual"], pred, packet["row_index"], packet["raw_record"])
                    self.logs["multiclass_best"].appendleft(entry)
                    results["multiclass_best"] = entry

                if "multiclass_nn" in self.models:
                    pred = self.models["multiclass_nn"].predict_from_row(packet["row"])
                    entry = self.build_entry(packet["actual"], pred, packet["row_index"], packet["raw_record"])
                    self.logs["multiclass_nn"].appendleft(entry)
                    results["multiclass_nn"] = entry
        except Exception as e:
            errors["multiclass"] = str(e)

        try:
            if "class33" in self.models and "class33" in self.datasets:
                packet = self.datasets["class33"].next_row()
                pred = self.models["class33"].predict_from_row(packet["row"])
                entry = self.build_entry(packet["actual"], pred, packet["row_index"], packet["raw_record"])
                self.logs["class33"].appendleft(entry)
                results["class33"] = entry
        except Exception as e:
            errors["class33"] = str(e)

        return {"results": results, "errors": errors}

    def reset(self):
        for k in self.logs:
            self.logs[k].clear()
        for d in self.datasets.values():
            d.index = 0

    def state(self):
        return {
            "logs": {
                key: [to_python(item) for item in value]
                for key, value in self.logs.items()
            }
        }


runtime = Runtime()
load_errors = runtime.load_all()

app = FastAPI(title="Cyber Demo Backend")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "errors": load_errors,
    }


@app.post("/tick")
def tick():
    return runtime.tick()


@app.post("/reset")
def reset():
    runtime.reset()
    return {"status": "reset"}


@app.get("/state")
def state():
    return runtime.state()


@app.get("/dashboard")
def dashboard():
    summary = {
        "binary": [],
        "multiclass": [],
        "class33": [],
    }

    binary_candidates = {
        "Logistic Regression": latest_run_metrics("logistic_regression_binary_"),
        "Random Forest": latest_run_metrics("random_forest_binary_"),
    }

    for model_name, item in binary_candidates.items():
        if item is None:
            continue
        run_dir, metrics = item
        summary["binary"].append({
            "model_name": model_name,
            "val_accuracy": extract_metric(metrics, ["val_accuracy", "validation_accuracy", "accuracy", "test_accuracy"]),
            "val_macro_f1": extract_metric(metrics, ["val_macro_f1", "validation_macro_f1", "macro_f1", "test_macro_f1"]),
            "test_accuracy": extract_metric(metrics, ["test_accuracy", "accuracy", "val_accuracy", "validation_accuracy"]),
            "test_macro_f1": extract_metric(metrics, ["test_macro_f1", "macro_f1", "val_macro_f1", "validation_macro_f1"]),
            "source": str(run_dir / "metrics.json"),
        })

    multiclass_candidates = {
        "Logistic Regression": latest_run_metrics("logistic_regression_multiclass"),
        "Random Forest": latest_run_metrics("random_forest_multiclass"),
    }

    for model_name, item in multiclass_candidates.items():
        if item is None:
            continue
        run_dir, metrics = item
        summary["multiclass"].append({
            "model_name": model_name,
            "val_accuracy": extract_metric(metrics, ["val_accuracy", "validation_accuracy", "accuracy", "test_accuracy"]),
            "val_macro_f1": extract_metric(metrics, ["val_macro_f1", "validation_macro_f1", "macro_f1", "test_macro_f1"]),
            "test_accuracy": extract_metric(metrics, ["test_accuracy", "accuracy", "val_accuracy", "validation_accuracy"]),
            "test_macro_f1": extract_metric(metrics, ["test_macro_f1", "macro_f1", "val_macro_f1", "validation_macro_f1"]),
            "source": str(run_dir / "metrics.json"),
        })

    nn_item = best_tuned_nn_metrics()
    if nn_item is not None:
        m = nn_item["metrics"]
        summary["multiclass"].append({
            "model_name": "Neural Network",
            "val_accuracy": extract_metric(m, ["val_accuracy", "validation_accuracy", "accuracy", "test_accuracy"]),
            "val_macro_f1": extract_metric(m, ["val_macro_f1", "validation_macro_f1", "macro_f1", "test_macro_f1"]),
            "test_accuracy": extract_metric(m, ["test_accuracy", "accuracy", "val_accuracy", "validation_accuracy"]),
            "test_macro_f1": extract_metric(m, ["test_macro_f1", "macro_f1", "val_macro_f1", "validation_macro_f1"]),
            "source": str(nn_item["metrics_path"]),
        })

    class33_candidates = {
        "Logistic Regression": latest_run_metrics("logistic_regression_33class"),
        "Random Forest": latest_run_metrics("random_forest_33class"),
    }

    for model_name, item in class33_candidates.items():
        if item is None:
            continue
        run_dir, metrics = item
        summary["class33"].append({
            "model_name": model_name,
            "val_accuracy": extract_metric(metrics, ["val_accuracy", "validation_accuracy", "accuracy", "test_accuracy"]),
            "val_macro_f1": extract_metric(metrics, ["val_macro_f1", "validation_macro_f1", "macro_f1", "test_macro_f1"]),
            "test_accuracy": extract_metric(metrics, ["test_accuracy", "accuracy", "val_accuracy", "validation_accuracy"]),
            "test_macro_f1": extract_metric(metrics, ["test_macro_f1", "macro_f1", "val_macro_f1", "validation_macro_f1"]),
            "source": str(run_dir / "metrics.json"),
        })

    return summary