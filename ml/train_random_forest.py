import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from utils import (
    load_config,
    load_dataset,
    get_features_and_target,
    validate_schema,
    replace_infinities_with_nan,
    compute_max_abs_finite_value,
    load_split_indices,
    select_split_data,
    compute_benign_fpr,
    make_run_dir,
    save_json,
    save_common_metadata,
)


# =========================================================
# 1) LOAD DATA AND CONFIG
# =========================================================
df = load_dataset("clean_sample.csv")
config = load_config("feature_list.json")

features, target = get_features_and_target(config)
validate_schema(df, features, target)

X = df[features].copy()
y = df[target].copy()

print("=" * 60)
print("🌲 STRONGER MODEL TRAINING - RANDOM FOREST")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Target column: {target}")
print(f"Number of features: {len(features)}")

print("\nOverall class distribution:")
print(y.value_counts())

# =========================================================
# 2) CLEAN INFINITE VALUES
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
# 3) LOAD FROZEN SPLIT
# =========================================================
train_idx, val_idx, test_idx, split_path = load_split_indices()

X_train, X_val, X_test, y_train, y_val, y_test = select_split_data(
    X, y, train_idx, val_idx, test_idx
)

print("\nLoaded reproducible split:")
print("  Train      :", X_train.shape, y_train.shape)
print("  Validation :", X_val.shape, y_val.shape)
print("  Test       :", X_test.shape, y_test.shape)

print("\nTrain class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())

# =========================================================
# 4) BUILD PIPELINE
#    No scaler here: Random Forest does not need scaling
# =========================================================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

# =========================================================
# 5) TRAIN MODEL
# =========================================================
pipeline.fit(X_train, y_train)

# =========================================================
# 6) VALIDATE
# =========================================================
y_val_pred = pipeline.predict(X_val)
y_val_proba = pipeline.predict_proba(X_val)

labels = list(pipeline.named_steps["model"].classes_)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")
cm = confusion_matrix(y_val, y_val_pred, labels=labels)
report_dict = classification_report(y_val, y_val_pred, output_dict=True)
report_text = classification_report(y_val, y_val_pred)

print("\n=== VALIDATION RESULTS ===")
print(f"Validation Accuracy : {val_accuracy:.6f}")
print(f"Validation Macro-F1 : {val_macro_f1:.6f}")

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

if "ATTACK" in labels:
    attack_idx = labels.index("ATTACK")
    attack_probs = y_val_proba[:, attack_idx]
    print(f"\nMean predicted ATTACK probability on validation set: {attack_probs.mean():.6f}")

# =========================================================
# 7) FEATURE IMPORTANCES
# =========================================================
rf_model = pipeline.named_steps["model"]

importance_df = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Feature Importances:")
print(importance_df.head(10).to_string(index=False))

# =========================================================
# 8) SAVE RUN ARTIFACTS
# =========================================================
run_dir = make_run_dir("random_forest")

model_path = os.path.join(run_dir, "model.joblib")
metrics_path = os.path.join(run_dir, "metrics.json")
report_path = os.path.join(run_dir, "classification_report.txt")
confusion_matrix_path = os.path.join(run_dir, "confusion_matrix.csv")
feature_importance_path = os.path.join(run_dir, "feature_importances.csv")

joblib.dump(pipeline, model_path)
save_common_metadata(run_dir, features, labels)

metrics = {
    "model": "RandomForestClassifier",
    "split_source": split_path,
    "validation_accuracy": float(val_accuracy),
    "validation_macro_f1": float(val_macro_f1),
    "false_positive_rate_benign": float(fpr_benign) if fpr_benign is not None else None,
    "class_order": labels,
    "train_size": int(len(X_train)),
    "validation_size": int(len(X_val)),
    "test_size": int(len(X_test)),
    "infinite_value_counts_by_feature": {str(k): int(v) for k, v in inf_counts.to_dict().items()},
    "max_abs_finite_numeric_value": float(max_abs_value),
    "classification_report": report_dict,
    "top_10_features": importance_df.head(10).to_dict(orient="records"),
}

save_json(metrics, metrics_path)

with open(report_path, "w") as f:
    f.write(report_text)

pd.DataFrame(cm, index=labels, columns=labels).to_csv(confusion_matrix_path)
importance_df.to_csv(feature_importance_path, index=False)

print("\nSaved run artifacts:")
print(f"  - {model_path}")
print(f"  - {metrics_path}")
print(f"  - {report_path}")
print(f"  - {confusion_matrix_path}")
print(f"  - {feature_importance_path}")
print(f"  - {os.path.join(run_dir, 'feature_order.json')}")
print(f"  - {os.path.join(run_dir, 'label_mapping.json')}")

print(f"\n✅ Random Forest run complete: {run_dir}")