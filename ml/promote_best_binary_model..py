from pathlib import Path
from datetime import datetime
import json
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"

FINAL_DIR = PROJECT_ROOT / "artifacts" / "final_binary"

MODEL_PREFIXES = {
    "logistic_regression_binary": "logistic_regression_binary_",
    "random_forest_binary": "random_forest_binary_",
}

REQUIRED_FILES = {
    "model.joblib": "best_model.joblib",
    "metrics.json": "best_model_metrics.json",
    "feature_order.json": "feature_order.json",
    "label_mapping.json": "label_mapping.json",
}

OPTIONAL_FILES = {
    "val_classification_report.txt": "val_classification_report.txt",
    "test_classification_report.txt": "test_classification_report.txt",
    "val_confusion_matrix.csv": "val_confusion_matrix.csv",
    "test_confusion_matrix.csv": "test_confusion_matrix.csv",
    "feature_importances.csv": "feature_importances.csv",
}


def fail(msg):
    raise RuntimeError(msg)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def find_latest_run(prefix: str) -> Path | None:
    candidates = []
    if not RUNS_ROOT.exists():
        return None

    for path in RUNS_ROOT.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def read_model_score(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        fail(f"Missing metrics file: {metrics_path}")

    metrics = load_json(metrics_path)

    score = metrics.get("val_macro_f1")
    if score is None:
        score = metrics.get("test_macro_f1")
    if score is None:
        fail(f"No macro-F1 found in {metrics_path}")

    accuracy = metrics.get("val_accuracy")
    if accuracy is None:
        accuracy = metrics.get("test_accuracy", 0.0)

    return metrics, float(score), float(accuracy)


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
    print("PROMOTE BEST BINARY MODEL")
    print("=" * 60)

    candidates = []

    for model_name, prefix in MODEL_PREFIXES.items():
        run_dir = find_latest_run(prefix)
        if run_dir is None:
            print(f"Skipping missing run for prefix: {prefix}")
            continue

        metrics, score, accuracy = read_model_score(run_dir)
        candidates.append((model_name, run_dir, metrics, score, accuracy))

        print(f"{model_name}:")
        print(f"  run       = {run_dir}")
        print(f"  macro-F1  = {score:.6f}")
        print(f"  accuracy  = {accuracy:.6f}")

    if not candidates:
        fail("No valid binary runs found.")

    candidates.sort(key=lambda x: (x[3], x[4]), reverse=True)

    best_model_name, best_run_dir, best_metrics, best_score, best_accuracy = candidates[0]

    print("\nSelected best binary model:")
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

    print("\nPromoted files into artifacts/final_binary:")
    for dst_name in REQUIRED_FILES.values():
        print(f"  - {FINAL_DIR / dst_name}")

    for dst_name in copied_optional:
        print(f"  - {FINAL_DIR / dst_name}")

    print(f"  - {FINAL_DIR / 'promotion_summary.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()