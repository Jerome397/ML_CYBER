import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def fail(message: str) -> None:
    print(f"\n❌ ERROR: {message}")
    sys.exit(1)


def load_config(config_path: str = "feature_list.json") -> dict:
    if not os.path.exists(config_path):
        fail(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        fail(f"Invalid JSON in {config_path}: {e}")


def load_dataset(csv_path: str = "clean_sample.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        fail(f"Dataset file not found: {csv_path}")

    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        fail(f"Could not read CSV file {csv_path}: {e}")


def get_features_and_target(config: dict) -> tuple[list, str]:
    if "target" not in config or "features" not in config:
        fail("feature_list.json must contain both 'target' and 'features' keys.")

    target = config["target"]
    features = config["features"]

    if not isinstance(features, list) or len(features) == 0:
        fail("'features' must be a non-empty list in feature_list.json.")

    return features, target


def validate_schema(df: pd.DataFrame, features: list, target: str) -> None:
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        fail(f"These declared features are missing from the CSV: {missing_features}")

    if target not in df.columns:
        fail(f"Target column '{target}' was not found in the dataset.")


def report_infinities(X: pd.DataFrame) -> pd.Series:
    inf_mask = np.isinf(X)
    inf_counts = pd.Series(inf_mask.sum(axis=0), index=X.columns)
    return inf_counts[inf_counts > 0].sort_values(ascending=False)


def replace_infinities_with_nan(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X_clean = X.copy()
    inf_counts = report_infinities(X_clean)

    if not inf_counts.empty:
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    return X_clean, inf_counts


def compute_max_abs_finite_value(X: pd.DataFrame) -> float:
    numeric_array = (
        X.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    return float(np.nanmax(np.abs(numeric_array)))


def load_split_indices(split_path: str = os.path.join("artifacts", "splits", "split_indices.npz")):
    if not os.path.exists(split_path):
        fail(f"Split file not found: {split_path}")

    try:
        split_data = np.load(split_path)
        train_idx = split_data["train_idx"]
        val_idx = split_data["val_idx"]
        test_idx = split_data["test_idx"]
        return train_idx, val_idx, test_idx,split_path
    except Exception as e:
        fail(f"Could not load split indices from {split_path}: {e}")


def select_split_data(X: pd.DataFrame, y: pd.Series, train_idx, val_idx, test_idx):
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()
    y_test = y.iloc[test_idx].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_benign_fpr(cm, labels):
    if len(labels) != 2:
        return None, "FPR on BENIGN traffic only computed for binary classification."

    if "BENIGN" not in labels:
        return None, "FPR on BENIGN traffic could not be computed because 'BENIGN' label was not found."

    benign_idx = labels.index("BENIGN")
    other_idx = 1 - benign_idx

    benign_total = cm[benign_idx, :].sum()
    benign_misclassified = cm[benign_idx, other_idx]

    if benign_total == 0:
        return None, "FPR on BENIGN traffic could not be computed because there are no BENIGN samples."

    fpr = benign_misclassified / benign_total
    return float(fpr), None


def make_run_dir(model_name: str, base_dir: str = os.path.join("artifacts", "runs")) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_common_metadata(run_dir: str, features: list, labels: list) -> None:
    feature_order_path = os.path.join(run_dir, "feature_order.json")
    label_mapping_path = os.path.join(run_dir, "label_mapping.json")

    save_json(features, feature_order_path)
    save_json({"classes": labels}, label_mapping_path)