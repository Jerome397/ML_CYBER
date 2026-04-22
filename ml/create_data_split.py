import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# HELPERS
# =========================================================
def fail(message: str) -> None:
    print(f"\n❌ ERROR: {message}")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        fail(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        fail(f"Invalid JSON in {config_path}: {e}")


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        fail(f"Dataset file not found: {csv_path}")

    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        fail(f"Could not read CSV file {csv_path}: {e}")


def report_infinities(X: pd.DataFrame) -> pd.Series:
    inf_mask = np.isinf(X)
    inf_counts = pd.Series(inf_mask.sum(axis=0), index=X.columns)
    return inf_counts[inf_counts > 0].sort_values(ascending=False)


def class_distribution(series: pd.Series) -> dict:
    counts = series.value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


# =========================================================
# PATHS / SETTINGS
# =========================================================
DATA_PATH = "clean_sample.csv"
CONFIG_PATH = "feature_list.json"

SPLITS_DIR = os.path.join("artifacts", "splits")
INDICES_PATH = os.path.join(SPLITS_DIR, "split_indices.npz")
SUMMARY_PATH = os.path.join(SPLITS_DIR, "split_summary.json")

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =========================================================
# 1) LOAD DATA AND CONFIG
# =========================================================
df = load_dataset(DATA_PATH)
config = load_config(CONFIG_PATH)

if "target" not in config or "features" not in config:
    fail("feature_list.json must contain both 'target' and 'features' keys.")

target = config["target"]
features = config["features"]

if not isinstance(features, list) or len(features) == 0:
    fail("'features' must be a non-empty list in feature_list.json.")

missing_features = [col for col in features if col not in df.columns]
if missing_features:
    fail(f"These declared features are missing from the CSV: {missing_features}")

if target not in df.columns:
    fail(f"Target column '{target}' was not found in the dataset.")

X = df[features].copy()
y = df[target].copy()

print("=" * 60)
print("🧩 CREATE REPRODUCIBLE TRAIN / VAL / TEST SPLIT")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Target column: {target}")
print(f"Number of features: {len(features)}")

print("\nOverall class distribution:")
print(y.value_counts())


# =========================================================
# 2) CHECK INFINITE VALUES
#    We do NOT save a cleaned CSV here.
#    We only record the issue so training scripts can clean
#    the same way every time before fitting.
# =========================================================
inf_counts = report_infinities(X)

if not inf_counts.empty:
    print("\n⚠️ Infinite values detected in these columns:")
    print(inf_counts)
else:
    print("\n✅ No infinite values detected.")

# Optional visibility into scale
numeric_array = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).to_numpy()
max_abs_value = np.nanmax(np.abs(numeric_array))
print(f"\nLargest absolute finite numeric value in X: {max_abs_value}")

# =========================================================
# 3) CREATE STRATIFIED SPLITS ON ROW INDICES
#    This freezes the exact rows used by every model later.
# =========================================================
indices = np.arange(len(df))

temp_size = int(len(df) * (VAL_RATIO + TEST_RATIO))   # 6000
val_size = int(len(df) * VAL_RATIO)                   # 3000
test_size = int(len(df) * TEST_RATIO)                 # 3000

train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices,
    y,
    test_size=temp_size,
    stratify=y,
    random_state=RANDOM_STATE
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=test_size,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

# Sort indices for readability / stability
train_idx = np.sort(train_idx)
val_idx = np.sort(val_idx)
test_idx = np.sort(test_idx)


# =========================================================
# 4) VERIFY SPLITS
# =========================================================
y_train_final = y.iloc[train_idx]
y_val_final = y.iloc[val_idx]
y_test_final = y.iloc[test_idx]

print("\nSplit sizes:")
print(f"  Train      : {len(train_idx)}")
print(f"  Validation : {len(val_idx)}")
print(f"  Test       : {len(test_idx)}")

print("\nTrain class distribution:")
print(y_train_final.value_counts())

print("\nValidation class distribution:")
print(y_val_final.value_counts())

print("\nTest class distribution:")
print(y_test_final.value_counts())

overlap_train_val = np.intersect1d(train_idx, val_idx)
overlap_train_test = np.intersect1d(train_idx, test_idx)
overlap_val_test = np.intersect1d(val_idx, test_idx)

if len(overlap_train_val) > 0 or len(overlap_train_test) > 0 or len(overlap_val_test) > 0:
    fail("Split overlap detected. Train/val/test must be disjoint.")

total_unique = len(np.unique(np.concatenate([train_idx, val_idx, test_idx])))
if total_unique != len(df):
    fail("Split coverage error: train/val/test do not cover the dataset exactly once.")

print("\n✅ Split integrity checks passed.")


# =========================================================
# 5) SAVE SPLITS
# =========================================================
os.makedirs(SPLITS_DIR, exist_ok=True)

np.savez_compressed(
    INDICES_PATH,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

summary = {
    "data_path": DATA_PATH,
    "config_path": CONFIG_PATH,
    "target": target,
    "num_features": int(len(features)),
    "num_rows": int(len(df)),
    "random_state": RANDOM_STATE,
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "overall_class_distribution": class_distribution(y),
    "train_class_distribution": class_distribution(y_train_final),
    "validation_class_distribution": class_distribution(y_val_final),
    "test_class_distribution": class_distribution(y_test_final),
    "infinite_value_counts_by_feature": {str(k): int(v) for k, v in inf_counts.to_dict().items()},
    "max_abs_numeric_value": float(max_abs_value)
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=4)

print("\nSaved:")
print(f"  - {INDICES_PATH}")
print(f"  - {SUMMARY_PATH}")

print("\n✅ Reproducible split creation complete.")