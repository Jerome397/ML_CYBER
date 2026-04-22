from pathlib import Path
import shutil
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
FINAL_NN_DIR = PROJECT_ROOT / "artifacts" / "final_multiclass_nn"

# Put your exact chosen tuned NN run folder name here
NN_RUN_NAME = "mlp_tuned_multiclass_v1_20260422_120037"

REQUIRED_FILES = {
    "model.joblib": "best_model.joblib",
    "feature_order.json": "feature_order.json",
    "label_mapping.json": "label_mapping.json",
}

OPTIONAL_FILES = {
    "metrics.json": "best_model_metrics.json",
    "promotion_ready_metrics.json": "promotion_ready_metrics.json",
    "best_params.json": "best_params.json",
    "training_config.json": "training_config.json",
    "val_classification_report.txt": "val_classification_report.txt",
    "test_classification_report.txt": "test_classification_report.txt",
    "val_confusion_matrix.csv": "val_confusion_matrix.csv",
    "test_confusion_matrix.csv": "test_confusion_matrix.csv",
}


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def main():
    source_dir = RUNS_ROOT / NN_RUN_NAME
    if not source_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {source_dir}")

    FINAL_NN_DIR.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in REQUIRED_FILES.items():
        src = source_dir / src_name
        dst = FINAL_NN_DIR / dst_name
        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")
        shutil.copy2(src, dst)

    copied_optional = []
    for src_name, dst_name in OPTIONAL_FILES.items():
        src = source_dir / src_name
        dst = FINAL_NN_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            copied_optional.append(dst_name)

    summary = {
        "source_run_dir": str(source_dir),
        "final_dir": str(FINAL_NN_DIR),
        "copied_optional_files": copied_optional,
    }
    save_json(summary, FINAL_NN_DIR / "promotion_summary.json")

    print("Done.")
    print(f"Promoted NN run to: {FINAL_NN_DIR}")


if __name__ == "__main__":
    main()