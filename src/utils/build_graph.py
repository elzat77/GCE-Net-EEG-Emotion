import os
import torch
import numpy as np
import mne


SEED_TO_MNE = {
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz",
    "CPZ": "CPz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
}

CHANNEL_ORDER = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]


def build_adjacency(sigma=0.15, top_k=8, save_path="static/adjacency_62.pt"):
    mne_channels = [SEED_TO_MNE.get(ch, ch) for ch in CHANNEL_ORDER]

    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()
    montage_positions = {ch: np.array(pos) for ch, pos in positions["ch_pos"].items()}

    o1_pos = montage_positions.get("O1", np.zeros(3))
    o2_pos = montage_positions.get("O2", np.zeros(3))
    tp7_pos = montage_positions.get("TP7", np.zeros(3))
    tp8_pos = montage_positions.get("TP8", np.zeros(3))
    montage_positions["CB1"] = (o1_pos + tp7_pos) / 2 + np.array([0, 0, -0.02])
    montage_positions["CB2"] = (o2_pos + tp8_pos) / 2 + np.array([0, 0, -0.02])

    coords = []
    for ch in mne_channels:
        if ch in montage_positions:
            coords.append(montage_positions[ch])
        else:
            raise KeyError(f"Channel {ch} not found in standard_1020 montage")

    coords = np.array(coords)
    V = coords.shape[0]

    dist_sq = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
    A = np.exp(-dist_sq / (2 * sigma ** 2))
    np.fill_diagonal(A, 0)

    A_topk = np.zeros_like(A)
    for i in range(V):
        idx = np.argpartition(A[i], -top_k)[-top_k:]
        A_topk[i, idx] = A[i, idx]

    A_topk = np.maximum(A_topk, A_topk.T)
    A_topk = A_topk + np.eye(V)

    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_topk.sum(axis=1)))
    A_norm = D_inv_sqrt @ A_topk @ D_inv_sqrt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(torch.from_numpy(A_norm).float(), save_path)
    print(f"Saved normalized adjacency matrix ({V}x{V}) to {save_path}")
    return A_norm


if __name__ == "__main__":
    build_adjacency()
