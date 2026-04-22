import json
import joblib
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIRS = {
    "binary": PROJECT_ROOT / "artifacts" / "final_binary",
    "multiclass_best": PROJECT_ROOT / "artifacts" / "final_multiclass",
    "multiclass_nn": PROJECT_ROOT / "artifacts" / "final_multiclass_nn",
    "33class": PROJECT_ROOT / "artifacts" / "final_33class",
}


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_model_bundle(model_key: str):
    if model_key not in MODEL_DIRS:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from: {list(MODEL_DIRS.keys())}")

    model_dir = MODEL_DIRS[model_key]

    model_path = model_dir / "best_model.joblib"
    feature_order_path = model_dir / "feature_order.json"
    label_mapping_path = model_dir / "label_mapping.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not feature_order_path.exists():
        raise FileNotFoundError(f"Missing feature_order.json: {feature_order_path}")
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"Missing label_mapping.json: {label_mapping_path}")

    model = joblib.load(model_path)
    feature_order = load_json(feature_order_path)
    label_mapping = load_json(label_mapping_path)

    idx_to_label = {int(v): k for k, v in label_mapping.items()}

    return {
        "model": model,
        "feature_order": feature_order,
        "label_mapping": label_mapping,
        "idx_to_label": idx_to_label,
        "model_dir": str(model_dir),
    }


def _predict_with_bundle(features: dict, bundle: dict):
    model = bundle["model"]
    feature_order = bundle["feature_order"]
    idx_to_label = bundle["idx_to_label"]

    missing_features = [f for f in feature_order if f not in features]
    extra_features = [f for f in features if f not in feature_order]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    row = {f: features[f] for f in feature_order}
    X = pd.DataFrame([row], columns=feature_order)

    pred_idx = model.predict(X)[0]
    pred_label = idx_to_label[int(pred_idx)]

    response = {
        "prediction_index": int(pred_idx),
        "prediction_label": pred_label,
        "ignored_extra_features": extra_features,
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        response["confidence"] = float(max(proba))
        response["probabilities"] = {
            idx_to_label[i]: float(proba[i]) for i in range(len(proba))
        }

    return response


def predict_best(features: dict, dataset_type: str):
    """
    dataset_type:
      - 'binary'
      - 'multiclass'
      - '33class'
    """
    if dataset_type == "binary":
        bundle = load_model_bundle("binary")
    elif dataset_type == "multiclass":
        bundle = load_model_bundle("multiclass_best")
    elif dataset_type == "33class":
        bundle = load_model_bundle("33class")
    else:
        raise ValueError("dataset_type must be one of: 'binary', 'multiclass', '33class'")

    result = _predict_with_bundle(features, bundle)
    result["dataset_type"] = dataset_type
    result["model_source"] = bundle["model_dir"]
    return result


def compare_multiclass_models(features: dict):
    """
    Runs both:
      - best promoted multiclass model
      - promoted multiclass neural network
    Returns both outputs side by side.
    """
    best_bundle = load_model_bundle("multiclass_best")
    nn_bundle = load_model_bundle("multiclass_nn")

    best_result = _predict_with_bundle(features, best_bundle)
    nn_result = _predict_with_bundle(features, nn_bundle)

    comparison = {
        "dataset_type": "multiclass",
        "best_model": {
            "model_source": best_bundle["model_dir"],
            **best_result,
        },
        "neural_network": {
            "model_source": nn_bundle["model_dir"],
            **nn_result,
        },
        "same_prediction": best_result["prediction_label"] == nn_result["prediction_label"],
    }

    return comparison


if __name__ == "__main__":
    pass