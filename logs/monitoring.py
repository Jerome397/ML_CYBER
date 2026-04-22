from collections import Counter

import numpy as np

# Simulated training means (replace later with real ones)
training_means = [0.5] * 39

logs = []


def log_event(features, prediction, confidence, timestamp):
    if len(features) != 39:
        raise ValueError("Expected 39 features")

    logs.append(
        {
            "features": features,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": timestamp,
        }
    )


def compute_alert_rate(logs):
    if len(logs) == 0:
        return 0

    alerts = [log for log in logs if log["prediction"] == "ATTACK"]
    return len(alerts) / len(logs)


def check_alert_rate(rate, threshold=0.4):
    if rate > threshold:
        return "⚠️ High alert rate"
    else:
        return "Normal"


def compute_prediction_distribution(logs):
    predictions = [log["prediction"] for log in logs]
    return dict(Counter(predictions))


def detect_attack_spike(distribution, threshold=0.5):
    total = sum(distribution.values())

    if total == 0:
        return "No data"

    attack_ratio = distribution.get("ATTACK", 0) / total

    if attack_ratio > threshold:
        return "⚠️ Attack spike detected"
    else:
        return "Normal"


def compute_feature_drift(logs, training_means):
    if len(logs) == 0:
        return []

    features_array = np.array([log["features"] for log in logs])
    current_means = np.mean(features_array, axis=0)

    drift_scores = abs(current_means - np.array(training_means))

    return drift_scores.tolist()


def detect_feature_drift(drift_scores, threshold=0.2):
    if any(d > threshold for d in drift_scores):
        return "⚠️ Feature drift detected"
    else:
        return "No drift"


def compute_risk(prediction, confidence, alert_rate):

    # Case 1: benign traffic
    if prediction == "BENIGN":
        return "LOW"

    # Case 2: system under heavy alerts → be more sensitive
    if alert_rate > 0.4:
        high_threshold = 0.7
    else:
        high_threshold = 0.9

    # Risk decision
    if confidence >= high_threshold:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def monitoring_pipeline(features, prediction, confidence, timestamp):

    # 1. log event
    log_event(features, prediction, confidence, timestamp)

    # 2. compute monitoring metrics
    alert_rate = compute_alert_rate(logs)
    distribution = compute_prediction_distribution(logs)
    drift = compute_feature_drift(logs, training_means)

    # 3. compute risk
    risk = compute_risk(prediction, confidence, alert_rate)

    return {
        "alert_rate": alert_rate,
        "distribution": distribution,
        "drift_sample": drift[:5],  # just first features
        "risk": risk,
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    logs.clear()

    result = monitoring_pipeline(
        features=[0] * 39,
        prediction="ATTACK",
        confidence=0.85,
        timestamp=123,
    )

    print(result)
