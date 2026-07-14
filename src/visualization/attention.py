import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..analysis.interpretability import CHANNEL_NAMES, BAND_NAMES


def plot_spatial_attention(attn_weights, save_path=None):
    weights = attn_weights.cpu().numpy() if hasattr(attn_weights, "cpu") else attn_weights
    if weights.ndim > 1:
        weights = weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(weights)), weights)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=5)
    ax.set_ylabel("Attention Weight")
    ax.set_title("Spatial Attention per Channel")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()


def plot_se_attention(se_weights, save_path=None):
    weights = se_weights.cpu().numpy() if hasattr(se_weights, "cpu") else se_weights
    weights = weights.squeeze()

    if len(weights) > 16:
        weights = weights[:16]

    fig, ax = plt.subplots(figsize=(8, 2))
    sns.heatmap(weights.reshape(1, -1), annot=True, fmt=".2f", cmap="Reds",
                xticklabels=False, yticklabels=False, ax=ax, cbar=True)
    ax.set_title("SE Channel Attention Weights")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()


def plot_attention_heatmap(attn_2d, xlabel="Time", ylabel="Channel", save_path=None):
    data = attn_2d.cpu().numpy() if hasattr(attn_2d, "cpu") else attn_2d

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, cmap="RdBu_r", center=0, ax=ax, cbar=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Attention Heatmap")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()
