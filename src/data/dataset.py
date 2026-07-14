import os
import numpy as np
import scipy.io as sio
import scipy.linalg
from torch.utils.data import Dataset


LABEL_SEQUENCE = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LABEL_MAP = {1: 0, 0: 1, -1: 2}


def compute_ea_params(data_dir, subject_ids, target_time=200, feature_key="de_LDS"):
    ea_params = {}
    for sid in subject_ids:
        band_features = {b: [] for b in range(5)}
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".mat") or fname == "label.mat":
                continue
            if int(fname.split("_")[0]) != sid:
                continue
            mat = sio.loadmat(os.path.join(data_dir, fname))
            for trial in range(1, 16):
                key = f"{feature_key}{trial}"
                if key not in mat:
                    continue
                X = mat[key].astype(np.float64)
                mean_t = X.mean(axis=1, keepdims=True)
                std_t = X.std(axis=1, keepdims=True) + 1e-8
                X = (X - mean_t) / std_t
                X = np.transpose(X, (2, 0, 1))
                T = X.shape[2]
                if T > target_time:
                    start = (T - target_time) // 2
                    X = X[:, :, start:start + target_time]
                elif T < target_time:
                    pad_width = target_time - T
                    pad_left = pad_width // 2
                    pad_right = pad_width - pad_left
                    X = np.pad(X, ((0, 0), (0, 0), (pad_left, pad_right)))
                for b in range(5):
                    band_features[b].append(X[b, :, :].T)
        if len(band_features[0]) < 2:
            continue
        band_params = {}
        for b in range(5):
            Fb = np.concatenate(band_features[b], axis=0)
            mu_b = Fb.mean(axis=0).astype(np.float32)
            Fc = Fb.astype(np.float64) - mu_b.astype(np.float64)
            Sigma = (Fc.T @ Fc) / (len(Fb) - 1)
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            threshold = eigvals.max() / 100.0
            eigvals = np.maximum(eigvals, threshold)
            W_b = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)
            band_params[b] = (mu_b, W_b)
        ea_params[sid] = band_params
    return ea_params


class SEED_DEDataset(Dataset):
    def __init__(self, data_dir, subject_ids, sessions=None, feature_key="de_LDS", target_time=200, ea_params=None):
        self.data_dir = data_dir
        self.subject_ids = sorted(subject_ids)
        self.sessions = sessions
        self.feature_key = feature_key
        self.target_time = target_time
        self.ea_params = ea_params
        self.samples = []
        self._cache = {}
        self._build_index()

    def _build_index(self):
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith(".mat") or fname == "label.mat":
                continue
            sid = int(fname.split("_")[0])
            if sid not in self.subject_ids:
                continue
            if self.sessions is not None:
                session = int(fname.split("_")[1].replace(".mat", ""))
                if session not in self.sessions:
                    continue
            filepath = os.path.join(self.data_dir, fname)
            mat = sio.loadmat(filepath)
            trial_data = {}
            for trial in range(1, 16):
                key = f"{self.feature_key}{trial}"
                if key not in mat:
                    continue
                trial_data[trial] = mat[key].astype(np.float32)
                self.samples.append((filepath, sid, trial))
            self._cache[filepath] = trial_data

    def _load_trial(self, filepath, trial):
        return self._cache[filepath][trial]

    def _normalize(self, X):
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mean) / std
        return X

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, subject_id, trial = self.samples[idx]
        X = self._load_trial(filepath, trial)
        X = self._normalize(X)
        X = np.transpose(X, (2, 0, 1))
        T = X.shape[2]
        if T > self.target_time:
            start = (T - self.target_time) // 2
            X = X[:, :, start:start + self.target_time]
        elif T < self.target_time:
            pad_width = self.target_time - T
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            X = np.pad(X, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant")
        if self.ea_params is not None and subject_id in self.ea_params:
            band_params = self.ea_params[subject_id]
            for b in range(X.shape[0]):
                mu_b, W_b = band_params[b]
                band_t = X[b, :, :].T.astype(np.float32)
                X[b, :, :] = ((band_t - mu_b) @ W_b.T).T
        y = LABEL_MAP[LABEL_SEQUENCE[trial - 1]]
        return (
            X.astype(np.float32),
            y,
            subject_id - 1,
            trial,
        )


class SEED_EEGDataset(Dataset):
    def __init__(self, data_dir, subject_ids, sessions=None, n_windows=200):
        self.data_dir = data_dir
        self.subject_ids = sorted(subject_ids)
        self.sessions = sessions
        self.n_windows = n_windows
        self.samples = []
        self._cache = {}
        self._build_index()

    def _find_eeg_prefix(self, mat):
        for k in mat.keys():
            if not k.startswith("__") and "_eeg1" in k:
                return k[:-1]
        for k in mat.keys():
            if not k.startswith("__") and "eeg" in k:
                return k.rsplit("eeg", 1)[0] + "eeg"
        return "ww_eeg"

    def _build_index(self):
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith(".mat") or fname == "label.mat":
                continue
            sid = int(fname.split("_")[0])
            if sid not in self.subject_ids:
                continue
            if self.sessions is not None:
                session = int(fname.split("_")[1].replace(".mat", ""))
                if session not in self.sessions:
                    continue
            filepath = os.path.join(self.data_dir, fname)
            mat = sio.loadmat(filepath)
            self._cache[filepath] = mat
            prefix = self._find_eeg_prefix(mat)
            for trial in range(1, 16):
                key = f"{prefix}{trial}"
                if key not in mat:
                    continue
                data = mat[key].astype(np.float32)
                T = data.shape[1]
                n_windows = (T - 1) // self.n_windows
                for w in range(n_windows):
                    start = w * self.n_windows
                    end = start + self.n_windows
                    self.samples.append((filepath, prefix, trial, start, end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, prefix, trial, start, end = self.samples[idx]
        sid = int(os.path.basename(filepath).split("_")[0])
        mat = self._cache[filepath]
        key = f"{prefix}{trial}"
        X = mat[key][:, start:end].astype(np.float32)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mean) / std
        X = X[np.newaxis, :, :]
        y = LABEL_MAP[LABEL_SEQUENCE[trial - 1]]
        return X, y, sid - 1, trial
