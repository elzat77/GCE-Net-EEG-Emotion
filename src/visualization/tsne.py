import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from . import get_emotion_labels


def plot_tsne(features, labels, title="t-SNE", save_path=None, color_by=None, color_labels=None):
    n = features.shape[0]
    perplexity = min(30, n - 1) if n > 1 else 1

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))

    if color_by is not None and color_labels is not None:
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                             c=color_labels, cmap="tab20", alpha=0.6, s=15)
        plt.colorbar(scatter, ax=ax)
        ax.set_title(f"{title} (colored by {color_by})")
    else:
        emotions = get_emotion_labels()
        colors = ["red", "gray", "blue"]
        for c in range(3):
            mask = labels == c
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                       c=colors[c], label=emotions[c], alpha=0.6, s=15)
        ax.legend()
        ax.set_title(title)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()

    return embedded
