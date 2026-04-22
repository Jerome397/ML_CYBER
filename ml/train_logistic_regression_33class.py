import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "runs" / "logistic_regression_33class"
DATASET_DIR = PROJECT_ROOT / "dataset" / "multiclass_33_v1"


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_33class_dataset(split_name: str):
    csv_path = DATASET_DIR / f"{split_name}.csv"
    feature_list_path = DATASET_DIR / "feature_list.json"
    label_mapping_path = DATASET_DIR / "label_mapping.json"

    df = pd.read_csv(csv_path)

    with open(feature_list_path, "r") as f:
        feature_list = json.load(f)

    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)

    X = df[feature_list].copy()
    y = df["Label"].map(label_mapping)

    if y.isna().any():
        missing_labels = df.loc[y.isna(), "Label"].unique().tolist()
        raise ValueError(f"Some labels were not found in label_mapping.json: {missing_labels}")

    y = y.astype(int)

    return {
        "X": X,
        "y": y,
        "feature_list": feature_list,
        "label_mapping": label_mapping
    }


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
    print("33-CLASS LOGISTIC REGRESSION TRAINING")
    print("=" * 60)

    train_data = load_33class_dataset("train")
    test_data = load_33class_dataset("test")

    X_full_train = train_data["X"].copy()
    y_full_train = train_data["y"].copy()

    X_test = test_data["X"].copy()
    y_test = test_data["y"].copy()

    train_inf_count = np.isinf(X_full_train.to_numpy()).sum()
    test_inf_count = np.isinf(X_test.to_numpy()).sum()

    print(f"Infinite values in train: {train_inf_count}")
    print(f"Infinite values in test : {test_inf_count}")

    X_full_train = X_full_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    feature_list = train_data["feature_list"]
    label_mapping = train_data["label_mapping"]

    idx_to_label = {v: k for k, v in label_mapping.items()}
    class_order = sorted(idx_to_label.keys())
    class_names = [idx_to_label[i] for i in class_order]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train,
        y_full_train,
        test_size=0.2,
        random_state=42,
        stratify=y_full_train
    )

    print(f"Train shape      : {X_train.shape}")
    print(f"Validation shape : {X_val.shape}")
    print(f"Test shape       : {X_test.shape}")

    print("\nTrain class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nValidation class distribution:")
    print(y_val.value_counts().sort_index())

    print("\nTest class distribution:")
    print(y_test.value_counts().sort_index())

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    val_results = evaluate_split("validation", y_val, y_val_pred, class_order, class_names)

    y_test_pred = pipeline.predict(X_test)
    test_results = evaluate_split("test", y_test, y_test_pred, class_order, class_names)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, ARTIFACTS_DIR / "model.joblib")
    val_results["confusion_matrix_df"].to_csv(ARTIFACTS_DIR / "val_confusion_matrix.csv")
    test_results["confusion_matrix_df"].to_csv(ARTIFACTS_DIR / "test_confusion_matrix.csv")

    with open(ARTIFACTS_DIR / "val_classification_report.txt", "w") as f:
        f.write(val_results["report_text"])

    with open(ARTIFACTS_DIR / "test_classification_report.txt", "w") as f:
        f.write(test_results["report_text"])

    save_json(feature_list, ARTIFACTS_DIR / "feature_order.json")
    save_json(label_mapping, ARTIFACTS_DIR / "label_mapping.json")

    metrics = {
        "model": "LogisticRegression",
        "dataset": "multiclass_33_v1",
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
        "test_weighted_f1": test_results["weighted_f1"]
    }

    save_json(metrics, ARTIFACTS_DIR / "metrics.json")

    print(f"\nSaved 33-class artifacts to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()