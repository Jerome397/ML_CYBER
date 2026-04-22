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
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# -----------------------------
# CHANGE THIS ONLY WHEN NEEDED
# -----------------------------
DATASET_NAME = "multiclass_v1"       # 7-class
# DATASET_NAME = "multiclass_33_v1"  # 33-class
# DATASET_NAME = "binary_v1"         # if you later have a binary dataset folder like this

DATASET_DIR = PROJECT_ROOT / "dataset" / DATASET_NAME

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"mlp_{DATASET_NAME}_{RUN_TIMESTAMP}"
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

    # Convert labels to integer IDs robustly
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

    return {
        "X": X,
        "y": y,
        "feature_list": feature_list,
        "label_mapping": label_mapping,
        "label_col": label_col,
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
    print("MLP / NEURAL NETWORK TRAINING")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Run folder: {ARTIFACTS_DIR}")

    train_data = load_dataset("train")
    test_data = load_dataset("test")

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

    idx_to_label = {int(v): k for k, v in label_mapping.items()}
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

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_results = evaluate_split("validation", y_val, y_val_pred, class_order, class_names)

    y_test_pred = model.predict(X_test)
    test_results = evaluate_split("test", y_test, y_test_pred, class_order, class_names)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

    val_results["confusion_matrix_df"].to_csv(ARTIFACTS_DIR / "val_confusion_matrix.csv")
    test_results["confusion_matrix_df"].to_csv(ARTIFACTS_DIR / "test_confusion_matrix.csv")

    with open(ARTIFACTS_DIR / "val_classification_report.txt", "w") as f:
        f.write(val_results["report_text"])

    with open(ARTIFACTS_DIR / "test_classification_report.txt", "w") as f:
        f.write(test_results["report_text"])

    save_json(feature_list, ARTIFACTS_DIR / "feature_order.json")
    save_json(label_mapping, ARTIFACTS_DIR / "label_mapping.json")

    training_config = {
        "model": "MLPClassifier",
        "dataset": DATASET_NAME,
        "run_name": RUN_NAME,
        "hidden_layer_sizes": [128, 64],
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": 256,
        "learning_rate_init": 0.001,
        "max_iter": 100,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "random_state": 42,
    }
    save_json(training_config, ARTIFACTS_DIR / "training_config.json")

    metrics = {
        "model": "MLPClassifier",
        "dataset": DATASET_NAME,
        "run_name": RUN_NAME,
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
    }
    save_json(metrics, ARTIFACTS_DIR / "metrics.json")
    save_json(metrics, ARTIFACTS_DIR / "promotion_ready_metrics.json")

    print(f"\nSaved neural network artifacts to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()