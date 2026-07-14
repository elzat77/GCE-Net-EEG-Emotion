import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from . import get_emotion_labels


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=True):
    emotions = get_emotion_labels()
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=emotions, yticklabels=emotions, ax=ax,
                vmin=0, vmax=1 if normalize else None)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()
    return cm


def plot_paired_confusion_bar(cm, save_path=None):
    emotions = get_emotion_labels()
    n = cm.shape[0]
    pairs, rates = [], []

    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append(f"{emotions[i]}\u2192{emotions[j]}")
                rates.append(cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(pairs)), rates, color="steelblue")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Confusion Rate")
    ax.set_title("Pairwise Confusion Rates")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()
