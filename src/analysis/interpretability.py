import numpy as np
import matplotlib.pyplot as plt
from ..visualization import get_emotion_labels


CHANNEL_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]

BAND_NAMES = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]


def channel_importance(attn_weights_list, save_path=None):
    if isinstance(attn_weights_list, list):
        all_attn = np.stack([w.cpu().numpy() if hasattr(w, "cpu") else w for w in attn_weights_list])
        avg_attn = all_attn.mean(axis=0)
    else:
        avg_attn = attn_weights_list

    sorted_idx = np.argsort(avg_attn)[::-1]
    sorted_names = [CHANNEL_NAMES[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(sorted_idx)), avg_attn[sorted_idx])
    ax.set_xticks(range(len(sorted_idx)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=6)
    ax.set_ylabel("Attention Weight")
    ax.set_title("Channel Importance (Spatial Attention)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()

    return sorted_names, avg_attn[sorted_idx]


def frequency_band_contribution(temp_conv_weights, save_path=None):
    weights = temp_conv_weights.cpu().numpy() if hasattr(temp_conv_weights, "cpu") else temp_conv_weights
    if weights.ndim > 1:
        weights = weights.squeeze()

    band_weights = np.abs(weights).mean(axis=tuple(range(1, weights.ndim)))
    band_pct = band_weights / band_weights.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    ax.bar(BAND_NAMES, band_pct, color=colors)
    ax.set_ylabel("Contribution (%)")
    ax.set_title("Frequency Band Contribution")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()

    return {BAND_NAMES[i]: float(band_pct[i]) for i in range(len(BAND_NAMES))}


def brain_topography(channel_weights, save_path=None):
    import mne

    mne_ch_names = []
    mne_map = {
        "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
        "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz",
        "CPZ": "CPz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
    }
    for ch in CHANNEL_NAMES:
        mne_ch_names.append(mne_map.get(ch, ch))

    info = mne.create_info(mne_ch_names, sfreq=200, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    evoked = mne.EvokedArray(channel_weights.reshape(62, 1), info)
    fig = evoked.plot_topomap(times=[0], show=False, ch_type="eeg", sensors=False)
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()
