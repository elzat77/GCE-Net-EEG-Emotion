import numpy as np
import matplotlib.pyplot as plt
import mne
from ..analysis.interpretability import CHANNEL_NAMES


MNE_CHANNEL_MAP = {
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz",
    "CPZ": "CPz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
}


def plot_topomap(channel_values, save_path=None, title="Brain Topography"):
    if isinstance(channel_values, list):
        channel_values = np.array(channel_values)
    if channel_values.ndim > 1:
        channel_values = channel_values.squeeze()

    mne_channels = [MNE_CHANNEL_MAP.get(ch, ch) for ch in CHANNEL_NAMES]

    info = mne.create_info(mne_channels, sfreq=200, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    evoked = mne.EvokedArray(channel_values.reshape(62, 1), info)
    fig = evoked.plot_topomap(times=[0], show=False, ch_type="eeg", sensors=False)
    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()


def plot_multi_topomap(channel_values_list, titles=None, save_path=None, n_cols=3):
    n = len(channel_values_list)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.atleast_1d(axes).flatten()

    mne_channels = [MNE_CHANNEL_MAP.get(ch, ch) for ch in CHANNEL_NAMES]
    info = mne.create_info(mne_channels, sfreq=200, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    for i, values in enumerate(channel_values_list):
        if i >= len(axes):
            break
        vals = values.squeeze()
        evoked = mne.EvokedArray(vals.reshape(62, 1), info)
        evoked.plot_topomap(times=[0], axes=axes[i], show=False, ch_type="eeg", sensors=False)
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.close()
