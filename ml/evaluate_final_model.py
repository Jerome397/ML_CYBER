import os
import joblib
import json
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from utils import (
    fail,
    load_config,
    load_dataset,
    get_features_and_target,
    validate_schema,
    replace_infinities_with_nan,
    compute_max_abs_finite_value,
    load_split_indices,
    select_split_data,
    compute_benign_fpr,
    save_json,
)


# =========================================================
# PATHS
# =========================================================
FINAL_DIR = os.path.join("artifacts", "final")
MODEL_PATH = os.path.join(FINAL_DIR, "best_model.joblib")
FEATURE_ORDER_PATH = os.path.join(FINAL_DIR, "feature_order.json")
LABEL_MAPPING_PATH = os.path.join(FINAL_DIR, "label_mapping.json")

TEST_METRICS_PATH = os.path.join(FINAL_DIR, "test_metrics.json")
TEST_REPORT_PATH = os.path.join(FINAL_DIR, "test_classification_report.txt")
TEST_CONFUSION_MATRIX_PATH = os.path.join(FINAL_DIR, "test_confusion_matrix.csv")


# =========================================================
# 1) CHECK FINAL ARTIFACTS EXIST
# =========================================================
required_paths = [
    MODEL_PATH,
    FEATURE_ORDER_PATH,
    LABEL_MAPPING_PATH,
]

for path in required_paths:
    if not os.path.exists(path):
        fail(f"Required final artifact not found: {path}")


# =========================================================
# 2) LOAD FINAL MODEL
# =========================================================
pipeline = joblib.load(MODEL_PATH)

# Try to detect final estimator name cleanly
if hasattr(pipeline, "named_steps") and len(pipeline.named_steps) > 0:
    final_step_name = list(pipeline.named_steps.keys())[-1]
    final_estimator = pipeline.named_steps[final_step_name]
    model_name = final_estimator.__class__.__name__
else:
    final_estimator = pipeline
    model_name = pipeline.__class__.__name__


# =========================================================
# 3) LOAD DATA AND CONFIG
# =========================================================
df = load_dataset("clean_sample.csv")
config = load_config("feature_list.json")

features, target = get_features_and_target(config)
validate_schema(df, features, target)

with open(FEATURE_ORDER_PATH, "r") as f:
    saved_features = json.load(f)

with open(LABEL_MAPPING_PATH, "r") as f:
    saved_label_mapping = json.load(f)

saved_labels = saved_label_mapping.get("classes")

if saved_features != features:
    fail("Feature mismatch between final model artifacts and current feature_list.json")

X = df[features].copy()
y = df[target].copy()

print("=" * 60)
print("🧪 FINAL MODEL TEST EVALUATION")
print("=" * 60)
print(f"Model path   : {MODEL_PATH}")
print(f"Model type   : {model_name}")
print(f"Dataset shape: {df.shape}")
print(f"Target column: {target}")
print(f"Num features : {len(features)}")


# =========================================================
# 4) CLEAN INFINITE VALUES
# =========================================================
X, inf_counts = replace_infinities_with_nan(X)

if not inf_counts.empty:
    print("\n⚠️ Infinite values detected and replaced with NaN:")
    print(inf_counts)
else:
    print("\n✅ No infinite values detected.")

max_abs_value = compute_max_abs_finite_value(X)
print(f"\nLargest absolute finite numeric value in X: {max_abs_value}")


# =========================================================
# 5) LOAD FROZEN SPLIT AND SELECT TEST SET ONLY
# =========================================================
train_idx, val_idx, test_idx, split_path = load_split_indices()

_, _, X_test, _, _, y_test = select_split_data(
    X, y, train_idx, val_idx, test_idx
)

print("\nLoaded frozen split for final evaluation:")
print("  Test :", X_test.shape, y_test.shape)

print("\nTest class distribution:")
print(y_test.value_counts())


# =========================================================
# 6) PREDICT ON TEST SET
# =========================================================
y_test_pred = pipeline.predict(X_test)

if hasattr(pipeline, "predict_proba"):
    y_test_proba = pipeline.predict_proba(X_test)
else:
    y_test_proba = None

if hasattr(final_estimator, "classes_"):
    labels = list(final_estimator.classes_)
else:
    labels = sorted(y_test.unique().tolist())


if saved_labels is not None and list(saved_labels) != labels:
    fail("Label mismatch between final model artifacts and loaded model classes")


# =========================================================
# 7) EVALUATE
# =========================================================
test_accuracy = accuracy_score(y_test, y_test_pred)
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
cm = confusion_matrix(y_test, y_test_pred, labels=labels)
report_dict = classification_report(y_test, y_test_pred, output_dict=True)
report_text = classification_report(y_test, y_test_pred)

print("\n=== TEST RESULTS ===")
print(f"Test Accuracy : {test_accuracy:.6f}")
print(f"Test Macro-F1 : {test_macro_f1:.6f}")

print("\nModel class order:")
print(labels)

print("\nClassification Report:")
print(report_text)

print("\nConfusion Matrix [rows=true, cols=pred]:")
print(pd.DataFrame(cm, index=labels, columns=labels))

fpr_benign, fpr_note = compute_benign_fpr(cm, labels)
if fpr_benign is not None:
    print(f"\nFalse Positive Rate on BENIGN traffic: {fpr_benign:.6f}")
else:
    print(f"\nNote: {fpr_note}")

mean_attack_probability = None
if y_test_proba is not None and "ATTACK" in labels:
    attack_idx = labels.index("ATTACK")
    mean_attack_probability = float(y_test_proba[:, attack_idx].mean())
    print(f"\nMean predicted ATTACK probability on test set: {mean_attack_probability:.6f}")


# =========================================================
# 8) SAVE TEST RESULTS
# =========================================================
metrics = {
    "evaluation_split": "test",
    "split_source": split_path,
    "model_path": MODEL_PATH,
    "model_type": model_name,
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(test_macro_f1),
    "false_positive_rate_benign": float(fpr_benign) if fpr_benign is not None else None,
    "class_order": labels,
    "test_size": int(len(X_test)),
    "test_class_distribution": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
    "infinite_value_counts_by_feature": {str(k): int(v) for k, v in inf_counts.to_dict().items()},
    "max_abs_finite_numeric_value": float(max_abs_value),
    "classification_report": report_dict,
    "mean_attack_probability": mean_attack_probability,
}

save_json(metrics, TEST_METRICS_PATH)

with open(TEST_REPORT_PATH, "w") as f:
    f.write(report_text)

pd.DataFrame(cm, index=labels, columns=labels).to_csv(TEST_CONFUSION_MATRIX_PATH)

print("\nSaved final test artifacts:")
print(f"  - {TEST_METRICS_PATH}")
print(f"  - {TEST_REPORT_PATH}")
print(f"  - {TEST_CONFUSION_MATRIX_PATH}")

print("\n✅ Final test evaluation complete.")