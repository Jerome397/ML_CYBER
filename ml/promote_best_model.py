import os
import json
import shutil
from datetime import datetime

from utils import fail, save_json


# =========================================================
# SETTINGS
# =========================================================
RUNS_DIR = os.path.join("artifacts", "runs")
FINAL_DIR = os.path.join("artifacts", "final_binary")

MODEL_PREFIXES = [
    "logistic_regression",
    "random_forest",
]

REQUIRED_FILES = {
    "model.joblib": "best_model.joblib",
    "metrics.json": "best_model_metrics.json",
    "feature_order.json": "feature_order.json",
    "label_mapping.json": "label_mapping.json",
}

OPTIONAL_FILES = {
    "classification_report.txt": "classification_report.txt",
    "confusion_matrix.csv": "confusion_matrix.csv",
    "feature_importances.csv": "feature_importances.csv",
}


# =========================================================
# HELPERS
# =========================================================
def find_latest_run(runs_dir: str, model_prefix: str) -> str:
    if not os.path.exists(runs_dir):
        fail(f"Runs directory not found: {runs_dir}")

    candidates = []
    for name in os.listdir(runs_dir):
        full_path = os.path.join(runs_dir, name)

        if not os.path.isdir(full_path) or not name.startswith(model_prefix + "_"):
            continue

        timestamp_str = name[len(model_prefix) + 1:]
        try:
            run_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            candidates.append((run_dt, full_path))
        except ValueError:
            continue

    if not candidates:
        fail(f"No valid run folders found for prefix '{model_prefix}'")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_score(metrics: dict) -> float:
    possible_keys = [
        "val_macro_f1",
        "test_macro_f1",
        "macro_f1",
        "validation_macro_f1",
        "best_val_macro_f1"
    ]

    for key in possible_keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])

    fail(f"No macro-F1 found. Available keys: {list(metrics.keys())}")


def get_accuracy(metrics: dict) -> float:
    possible_keys = [
        "val_accuracy",
        "test_accuracy",
        "accuracy",
        "validation_accuracy",
        "best_val_accuracy"
    ]

    for key in possible_keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])

    return 0.0


def copy_required_files(source_dir: str, final_dir: str):
    for src_name, dst_name in REQUIRED_FILES.items():
        src_path = os.path.join(source_dir, src_name)
        dst_path = os.path.join(final_dir, dst_name)

        if not os.path.exists(src_path):
            fail(f"Missing required file: {src_path}")

        shutil.copy2(src_path, dst_path)


def copy_optional_files(source_dir: str, final_dir: str):
    copied = []
    for src_name, dst_name in OPTIONAL_FILES.items():
        src_path = os.path.join(source_dir, src_name)
        dst_path = os.path.join(final_dir, dst_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied.append(dst_name)

    return copied


# =========================================================
# MAIN
# =========================================================
print("=" * 60)
print("PROMOTE BEST MODEL AUTOMATICALLY (BINARY)")
print("=" * 60)

candidates = []

for model_prefix in MODEL_PREFIXES:
    run_dir = find_latest_run(RUNS_DIR, model_prefix)
    metrics_path = os.path.join(run_dir, "metrics.json")

    if not os.path.exists(metrics_path):
        fail(f"Missing metrics.json in {run_dir}")

    metrics = load_json(metrics_path)

    print(f"\nAvailable metric keys for {model_prefix}: {list(metrics.keys())}")

    score = get_score(metrics)
    acc = get_accuracy(metrics)

    candidates.append({
        "model": model_prefix,
        "run_dir": run_dir,
        "macro_f1": score,
        "accuracy": acc,
    })

    print(f"{model_prefix}:")
    print(f"  run       = {run_dir}")
    print(f"  macro-F1  = {score:.6f}")
    print(f"  accuracy  = {acc:.6f}")

# Choose best model
candidates.sort(key=lambda x: (x["macro_f1"], x["accuracy"]), reverse=True)
best = candidates[0]

print("\nSelected best model:")
print(f"  model     = {best['model']}")
print(f"  run       = {best['run_dir']}")
print(f"  macro-F1  = {best['macro_f1']:.6f}")

# =========================================================
# CREATE FINAL DIR (SAFE)
# =========================================================
os.makedirs(FINAL_DIR, exist_ok=True)

# =========================================================
# COPY FILES
# =========================================================
copy_required_files(best["run_dir"], FINAL_DIR)
copied_optional = copy_optional_files(best["run_dir"], FINAL_DIR)

# =========================================================
# SAVE SUMMARY
# =========================================================
summary = {
    "selected_model": best["model"],
    "source_run": best["run_dir"],
    "macro_f1": best["macro_f1"],
    "accuracy": best["accuracy"],
    "all_candidates": candidates,
    "promoted_at": datetime.now().isoformat(timespec="seconds"),
}

summary_path = os.path.join(FINAL_DIR, "promotion_summary.json")
save_json(summary, summary_path)

print("\nPromoted files:")
for name in REQUIRED_FILES.values():
    print(f"  - {os.path.join(FINAL_DIR, name)}")

for name in copied_optional:
    print(f"  - {os.path.join(FINAL_DIR, name)}")

print(f"  - {summary_path}")

print("\nBest model promotion complete.")