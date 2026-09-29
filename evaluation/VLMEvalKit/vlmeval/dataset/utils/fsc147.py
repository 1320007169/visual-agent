"""Numeric extraction and counting metrics without a judge model."""

import math
import re


def extract_count(prediction):
    text = str(prediction).strip()
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if answer:
        text = answer.group(1).strip()
    if not re.fullmatch(r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.0+)?", text):
        return None
    return int(text.replace(",", "").split(".")[0])


def counting_metrics(answers, predictions):
    if not answers or len(answers) != len(predictions):
        raise ValueError("FSC147 evaluation requires matching nonempty answers and predictions")
    extracted = [extract_count(prediction) for prediction in predictions]
    errors = [prediction - int(answer) for answer, prediction in zip(answers, extracted) if prediction is not None]
    valid = len(errors)
    total = len(answers)
    mae = sum(abs(error) for error in errors) / valid if valid else float("nan")
    rmse = math.sqrt(sum(error * error for error in errors) / valid) if valid else float("nan")
    # Incomplete predictions cannot yield a standard full-split MAE/RMSE.
    metrics = {
        "MAE": mae if valid == total else float("nan"),
        "RMSE": rmse if valid == total else float("nan"),
        "MAE_valid": mae, "RMSE_valid": rmse,
        "exact_accuracy": 100 * sum(error == 0 for error in errors) / total,
        "valid_rate": 100 * valid / total,
        "total_samples": total, "valid_samples": valid, "invalid_count": total - valid,
        "api_failed_count": sum("Failed to obtain answer via API" in str(p) for p in predictions),
    }
    return metrics, extracted
