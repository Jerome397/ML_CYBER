import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "runs" / f"random_forest_binary_{RUN_TIMESTAMP}"


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def find_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exist:\n" + "\n".join(str(p) for p in candidates))


def detect_label_column(df: pd.DataFrame) -> str:
    for c in ["Final_Label", "Label", "label", "Class", "class", "Target", "target"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find label column. Available columns: {list(df.columns)}")


def normalize_feature_list(raw):
    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ["feature_list", "features", "columns"]:
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return list(raw.keys())

    raise ValueError(f"Unsupported feature_list format: {type(raw)}")


def load_binary_dataset():
    csv_path = find_existing_path([
        PROJECT_ROOT / "clean_sample.csv",
        PROJECT_ROOT / "dataset" / "clean_sample.csv",
        PROJECT_ROOT / "dataset" / "binary_v1" / "clean_sample.csv",
        PROJECT_ROOT / "data" / "clean_sample.csv",
    ])

    feature_list_path = find_existing_path([
        PROJECT_ROOT / "feature_list.json",
        PROJECT_ROOT / "dataset" / "feature_list.json",
        PROJECT_ROOT / "dataset" / "binary_v1" / "feature_list.json",
        PROJECT_ROOT / "data" / "feature_list.json",
    ])

    df = pd.read_csv(csv_path)

    with open(feature_list_path, "r", encoding="utf-8") as f:
        raw_feature_list = json.load(f)

    feature_list = normalize_feature_list(raw_feature_list)

    label_col = detect_label_column(df)

    X = df[feature_list].copy()
    y_raw = df[label_col].copy()

    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int)
        unique_labels = sorted(pd.Series(y).dropna().unique().tolist())
        label_mapping = {str(lbl): int(lbl) for lbl in unique_labels}
    else:
        unique_labels = sorted(pd.Series(y_raw).dropna().astype(str).unique().tolist())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y = y_raw.astype(str).map(label_mapping).astype(int)

    return X, y, feature_list, label_mapping, label_col, str(csv_path)


def evaluate_split(name, y_true, y_pred, class_order, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n=== {name.upper()} RESULTS ===")
    print(f"Accuracy   : {acc:.6f}")
    print(f"Macro-F1   : {macro_f1:.6f}")
    print(f"Weighted-F1: {weighted_f1:.6f}")

    report_text = classification_report(
        y_true, y_pred, labels=class_order, target_names=class_names, zero_division=0
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
        "confusion_matrix_df": cm_df,
    }


def main():
    print("=" * 60)
    print("BINARY RANDOM FOREST TRAINING")
    print("=" * 60)

    X, y, feature_list, label_mapping, label_col, csv_path = load_binary_dataset()

    inf_count = np.isinf(X.to_numpy()).sum()
    print(f"Infinite values in dataset: {inf_count}")

    X = X.replace([np.inf, -np.inf], np.nan)

    idx_to_label = {int(v): k for k, v in label_mapping.items()}
    class_order = sorted(idx_to_label.keys())
    class_names = [idx_to_label[i] for i in class_order]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    val_results = evaluate_split("validation", y_val, y_val_pred, class_order, class_names)

    y_test_pred = pipeline.predict(X_test)
    test_results = evaluate_split("test", y_test, y_test_pred, class_order, class_names)

    rf_model = pipeline.named_steps["model"]
    importance_df = pd.DataFrame({
        "feature": feature_list,
        "importance": rf_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, ARTIFACTS_DIR / "model.joblib")
    val_results["confusion_matrix_df"].to_csv(ARTIFACTS_DIR / "val_confusion_matrix.csv")
    test_results["confusion_matrix_df"].to_csv(ARTIFACTS_DIR / "test_confusion_matrix.csv")
    importance_df.to_csv(ARTIFACTS_DIR / "feature_importances.csv", index=False)

    with open(ARTIFACTS_DIR / "val_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(val_results["report_text"])

    with open(ARTIFACTS_DIR / "test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(test_results["report_text"])

    save_json(feature_list, ARTIFACTS_DIR / "feature_order.json")
    save_json(label_mapping, ARTIFACTS_DIR / "label_mapping.json")

    metrics = {
        "model": "RandomForestClassifier",
        "dataset": "binary",
        "run_name": ARTIFACTS_DIR.name,
        "source_csv": csv_path,
        "label_column": label_col,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "val_accuracy": val_results["accuracy"],
        "val_macro_f1": val_results["macro_f1"],
        "val_weighted_f1": val_results["weighted_f1"],
        "test_accuracy": test_results["accuracy"],
        "test_macro_f1": test_results["macro_f1"],
        "test_weighted_f1": test_results["weighted_f1"],
        "top_10_features": importance_df.head(10).to_dict(orient="records"),
    }

    save_json(metrics, ARTIFACTS_DIR / "metrics.json")

    print(f"\nSaved binary random forest artifacts to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()