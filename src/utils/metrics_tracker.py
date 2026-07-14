import json
import os
import numpy as np


class MetricsTracker:
    def __init__(self):
        self.folds = {}

    def add_fold(self, fold_idx, metrics):
        self.folds[str(fold_idx)] = metrics

    def _summarize(self):
        if not self.folds:
            return {}

        all_keys = set()
        for m in self.folds.values():
            all_keys.update(m.keys())

        summary = {}
        for key in sorted(all_keys):
            values = [self.folds[f].get(key, np.nan) for f in self.folds]
            arr = np.array(values, dtype=float)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                summary[key] = {"mean": None, "std": None}
            else:
                summary[key] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                    "values": valid.tolist(),
                }
        return summary

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output = {
            "folds": self.folds,
            "summary": self._summarize(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
