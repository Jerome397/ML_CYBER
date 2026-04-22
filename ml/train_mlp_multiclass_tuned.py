import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATASET_NAME = "multiclass_v1"
DATASET_DIR = PROJECT_ROOT / "dataset" / DATASET_NAME

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"mlp_tuned_{DATASET_NAME}_{RUN_TIMESTAMP}"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "runs" / RUN_NAME


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def detect_label_column(df: pd.DataFrame) -> str:
    if "Final_Label" in df.columns:
        return "Final_Label"
    if "Label" in df.columns:
        return "Label"
    raise ValueError(f"Could not find label column. Available columns: {list(df.columns)}")


def load_dataset(split_name: str):
    csv_path = DATASET_DIR / f"{split_name}.csv"
    feature_list_path = DATASET_DIR / "feature_list.json"
    label_mapping_path = DATASET_DIR / "label_mapping.json"

    df = pd.read_csv(csv_path)

    with open(feature_list_path, "r") as f:
        feature_list = json.load(f)

    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)

    label_col = detect_label_column(df)

    X = df[feature_list].copy()
    y_raw = df[label_col].copy()

    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int)
    else:
        y = y_raw.map(label_mapping)
        if y.isna().any():
            missing_labels = sorted(df.loc[y.isna(), label_col].astype(str).unique().tolist())
            raise ValueError(
                f"Some labels were not found in label_mapping.json: {missing_labels}"
            )
        y = y.astype(int)

    return X, y, feature_list, label_mapping


def evaluate_split(name, y_true, y_pred, class_order, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n=== {name.upper()} RESULTS ===")
    print(f"Accuracy   : {acc:.6f}")
    print(f"Macro-F1   : {macro_f1:.6f}")
    print(f"Weighted-F1: {weighted_f1:.6f}")

    report_text = classification_report(
        y_true,
        y_pred,
        labels=class_order,
        target_names=class_names,
        zero_division=0
    )
    print("\nClassification Report:")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "report_text": report_text,
        "confusion_matrix_df": cm_df
    }


def main():
    print("=" * 60)
    print("TUNED NEURAL NETWORK TRAINING")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Run folder: {ARTIFACTS_DIR}")

    X_full, y_full, feature_list, label_mapping = load_dataset("train")
    X_test, y_test, _, _ = load_dataset("test")

    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    idx_to_label = {int(v): k for k, v in label_mapping.items()}
    class_order = sorted(idx_to_label.keys())
    class_names = [idx_to_label[i] for i in class_order]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        stratify=y_full,
        random_state=42
    )

    print(f"Train shape      : {X_train.shape}")
    print(f"Validation shape : {X_val.shape}")
    print(f"Test shape       : {X_test.shape}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            max_iter=150,
            early_stopping=True,
            random_state=42
        ))
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.0005, 0.001, 0.01],
        "mlp__batch_size": [128, 256],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_macro",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\nBest parameters:")
    print(grid.best_params_)
    print(f"Best CV macro-F1: {grid.best_score_:.6f}")

    y_val_pred = best_model.predict(X_val)
    val_results = evaluate_split("validation", y_val, y_val_pred, class_order, class_names)

    y_test_pred = best_model.predict(X_test)
    test_results = evaluate_split("test", y_test, y_test_pred, class_order, class_names)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, ARTIFACTS_DIR / "model.joblib")

    val_results["confusion_matrix_df"].to_csv(ARTIFACTS_DIR / "val_confusion_matrix.csv")
    test_results["confusion_matrix_df"].to_csv(ARTIFACTS_DIR / "test_confusion_matrix.csv")

    with open(ARTIFACTS_DIR / "val_classification_report.txt", "w") as f:
        f.write(val_results["report_text"])

    with open(ARTIFACTS_DIR / "test_classification_report.txt", "w") as f:
        f.write(test_results["report_text"])

    save_json(feature_list, ARTIFACTS_DIR / "feature_order.json")
    save_json(label_mapping, ARTIFACTS_DIR / "label_mapping.json")
    save_json(grid.best_params_, ARTIFACTS_DIR / "best_params.json")

    training_config = {
        "model": "MLPClassifier",
        "dataset": DATASET_NAME,
        "run_name": RUN_NAME,
        "search_type": "GridSearchCV",
        "scoring": "f1_macro",
        "cv": 3,
        "param_grid": {
            "mlp__hidden_layer_sizes": ["(64,)", "(128,)", "(128, 64)", "(256, 128)"],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [0.0001, 0.001, 0.01],
            "mlp__learning_rate_init": [0.0005, 0.001, 0.01],
            "mlp__batch_size": [128, 256],
        },
        "best_params": grid.best_params_,
        "best_cv_macro_f1": float(grid.best_score_),
        "random_state": 42,
    }
    save_json(training_config, ARTIFACTS_DIR / "training_config.json")

    metrics = {
        "model": "MLPClassifier",
        "dataset": DATASET_NAME,
        "run_name": RUN_NAME,
        "best_params": grid.best_params_,
        "best_cv_macro_f1": float(grid.best_score_),
        "val_accuracy": val_results["accuracy"],
        "val_macro_f1": val_results["macro_f1"],
        "val_weighted_f1": val_results["weighted_f1"],
        "test_accuracy": test_results["accuracy"],
        "test_macro_f1": test_results["macro_f1"],
        "test_weighted_f1": test_results["weighted_f1"],
    }

    save_json(metrics, ARTIFACTS_DIR / "metrics.json")
    save_json(metrics, ARTIFACTS_DIR / "promotion_ready_metrics.json")

    print(f"\nSaved tuned model to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()