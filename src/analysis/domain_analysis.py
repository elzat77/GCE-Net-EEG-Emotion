import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ..visualization import get_emotion_labels


def domain_confusion_matrix(domain_preds, domain_labels, n_domains=14, save_path=None):
    cm = confusion_matrix(domain_labels, domain_preds, labels=range(n_domains))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(n_domains):
        for j in range(n_domains):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=6)
    ax.set_xlabel("Predicted Domain")
    ax.set_ylabel("True Domain")
    ax.set_title("Domain Confusion Matrix")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()
    return cm_norm


def domain_accuracy_curve(history, save_path=None):
    domain_acc = history.get("domain_acc", [])
    val_acc = history.get("val_acc", [])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(val_acc, "b-", label="Val Accuracy")
    ax1.set_ylabel("Val Accuracy", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(domain_acc, "r--", label="Domain Accuracy")
    ax2.set_ylabel("Domain Accuracy", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_xlabel("Epoch")
    ax1.set_title("Accuracy vs Domain Accuracy")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()


def tsne_by_domain(features, domain_labels, emotion_labels, save_path=None):
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0] - 1))
    embedded = tsne.fit_transform(features)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    emotions = get_emotion_labels()
    emo_colors = ["red", "gray", "blue"]
    for e in range(3):
        mask = emotion_labels == e
        ax1.scatter(embedded[mask, 0], embedded[mask, 1], c=emo_colors[e], label=emotions[e], alpha=0.5, s=10)
    ax1.set_title("t-SNE colored by Emotion")
    ax1.legend()

    scatter2 = ax2.scatter(embedded[:, 0], embedded[:, 1], c=domain_labels, cmap="tab20", alpha=0.5, s=10)
    ax2.set_title("t-SNE colored by Domain (Subject)")
    plt.colorbar(scatter2, ax=ax2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()

    return embedded
