from pathlib import Path
from datetime import datetime
import json
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUN_DIRS = {
    "logistic_regression_multiclass": PROJECT_ROOT / "artifacts" / "runs" / "logistic_regression_multiclass",
    "random_forest_multiclass": PROJECT_ROOT / "artifacts" / "runs" / "random_forest_multiclass",
}

FINAL_DIR = PROJECT_ROOT / "artifacts" / "final_multiclass"

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
    "val_classification_report.txt": "val_classification_report.txt",
    "test_classification_report.txt": "test_classification_report.txt",
    "val_confusion_matrix.csv": "val_confusion_matrix.csv",
    "test_confusion_matrix.csv": "test_confusion_matrix.csv",
}


def fail(msg):
    raise RuntimeError(msg)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def read_model_score(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        fail(f"Missing metrics file: {metrics_path}")

    metrics = load_json(metrics_path)

    possible_score_keys = [
        "val_macro_f1",
        "test_macro_f1",
        "macro_f1",
        "validation_macro_f1",
        "best_val_macro_f1",
    ]

    score = None
    for key in possible_score_keys:
        if key in metrics and metrics[key] is not None:
            score = float(metrics[key])
            break

    if score is None:
        fail(f"No macro-F1 found in {metrics_path}. Available keys: {list(metrics.keys())}")

    possible_accuracy_keys = [
        "val_accuracy",
        "test_accuracy",
        "accuracy",
        "validation_accuracy",
        "best_val_accuracy",
    ]

    accuracy = 0.0
    for key in possible_accuracy_keys:
        if key in metrics and metrics[key] is not None:
            accuracy = float(metrics[key])
            break

    return metrics, score, accuracy


def copy_files(source_dir: Path, final_dir: Path):
    for src_name, dst_name in REQUIRED_FILES.items():
        src_path = source_dir / src_name
        dst_path = final_dir / dst_name

        if not src_path.exists():
            fail(f"Required file missing: {src_path}")

        shutil.copy2(src_path, dst_path)

    copied_optional = []
    for src_name, dst_name in OPTIONAL_FILES.items():
        src_path = source_dir / src_name
        dst_path = final_dir / dst_name

        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied_optional.append(dst_name)

    return copied_optional


def main():
    print("=" * 60)
    print("PROMOTE BEST MULTICLASS MODEL")
    print("=" * 60)

    candidates = []

    for model_name, run_dir in RUN_DIRS.items():
        if not run_dir.exists():
            print(f"Skipping missing run folder: {run_dir}")
            continue

        metrics, score, accuracy = read_model_score(run_dir)
        candidates.append((model_name, run_dir, metrics, score, accuracy))

        print(f"{model_name}:")
        print(f"  run       = {run_dir}")
        print(f"  macro-F1  = {score:.6f}")
        print(f"  accuracy  = {accuracy:.6f}")

    if not candidates:
        fail("No valid multiclass run folders found.")

    candidates.sort(
        key=lambda x: (x[3], x[4]),
        reverse=True
    )

    best_model_name, best_run_dir, best_metrics, best_score, best_accuracy = candidates[0]

    print("\nSelected best multiclass model:")
    print(f"  Name      : {best_model_name}")
    print(f"  Run dir   : {best_run_dir}")
    print(f"  Macro-F1  : {best_score:.6f}")
    print(f"  Accuracy  : {best_accuracy:.6f}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    copied_optional = copy_files(best_run_dir, FINAL_DIR)

    summary = {
        "promoted_model_name": best_model_name,
        "source_run_dir": str(best_run_dir),
        "selection_metric": "macro_f1",
        "selected_score": best_score,
        "selected_accuracy": best_accuracy,
        "all_candidates": [
            {
                "model_name": model_name,
                "run_dir": str(run_dir),
                "macro_f1": score,
                "accuracy": accuracy,
            }
            for model_name, run_dir, metrics, score, accuracy in candidates
        ],
        "required_files_promoted": list(REQUIRED_FILES.values()),
        "optional_files_promoted": copied_optional,
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
    }

    save_json(summary, FINAL_DIR / "promotion_summary.json")

    print("\nPromoted files into artifacts/final_multiclass:")
    for dst_name in REQUIRED_FILES.values():
        print(f"  - {FINAL_DIR / dst_name}")

    for dst_name in copied_optional:
        print(f"  - {FINAL_DIR / dst_name}")

    print(f"  - {FINAL_DIR / 'promotion_summary.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()