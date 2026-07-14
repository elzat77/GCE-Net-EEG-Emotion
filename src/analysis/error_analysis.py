import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ..visualization import get_emotion_labels


def per_class_confusion_metrics(cm):
    n = cm.shape[0]
    emotions = get_emotion_labels()
    result = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                key = f"{emotions[i]}->{emotions[j]}"
                result[key] = float(cm[i, j] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
    return result


def hardest_class_pair(cm):
    n = cm.shape[0]
    emotions = get_emotion_labels()
    max_confusion = 0
    hardest_pair = (0, 0)
    for i in range(n):
        for j in range(n):
            if i != j:
                rate = cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0.0
                if rate > max_confusion:
                    max_confusion = rate
                    hardest_pair = (i, j)
    return {
        "pair": f"{emotions[hardest_pair[0]]}->{emotions[hardest_pair[1]]}",
        "confusion_rate": float(max_confusion),
    }


def error_case_summary(y_true, y_pred):
    errors = y_true != y_pred
    n_total = len(y_true)
    n_errors = errors.sum()
    return {
        "total_samples": int(n_total),
        "error_count": int(n_errors),
        "error_rate": float(n_errors / n_total) if n_total > 0 else 0.0,
    }
