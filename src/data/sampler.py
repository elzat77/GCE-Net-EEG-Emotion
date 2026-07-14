import random
import numpy as np
import torch
from torch.utils.data import Sampler


class MultiSubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, n_subjects_min=3, n_per_emotion_min=1, n_same_subject_emotion_max=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_subjects_min = n_subjects_min
        self.n_per_emotion_min = n_per_emotion_min
        self.n_same_subject_emotion_max = n_same_subject_emotion_max

        self.subject_to_indices = {}
        for idx in range(len(dataset)):
            _, y, sid, _ = dataset[idx]
            self.subject_to_indices.setdefault(sid, {}).setdefault(y, []).append(idx)

        self.subjects = sorted(self.subject_to_indices.keys())
        self.n_samples = len(dataset)

    def _build_batch(self):
        batch = []
        available_subjects = list(self.subjects)
        random.shuffle(available_subjects)

        selected_subjects = []
        for sid in available_subjects:
            if len(selected_subjects) >= self.n_subjects_min:
                break
            if sid not in self.subject_to_indices:
                continue
            subject_emotions = self.subject_to_indices[sid]
            if len(subject_emotions) < 3:
                continue
            selected_subjects.append(sid)

        if len(selected_subjects) < self.n_subjects_min:
            return None

        candidate_indices = []
        for sid in selected_subjects:
            for emotion, indices in self.subject_to_indices[sid].items():
                candidate_indices.append((sid, emotion, indices))

        random.shuffle(candidate_indices)

        emotion_counts = {0: 0, 1: 0, 2: 0}
        subject_emotion_used = {}

        for sid, emotion, indices in candidate_indices:
            if len(batch) >= self.batch_size:
                break
            for idx in indices:
                if len(batch) >= self.batch_size:
                    break
                se_key = (sid, emotion)
                if subject_emotion_used.get(se_key, 0) >= self.n_same_subject_emotion_max:
                    break
                if idx not in batch:
                    batch.append(idx)
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    subject_emotion_used[se_key] = subject_emotion_used.get(se_key, 0) + 1

        if len(batch) < self.batch_size:
            return None

        has_all_emotions = all(emotion_counts.get(e, 0) >= self.n_per_emotion_min for e in range(3))
        n_subjects_in_batch = len(set(self.dataset[i][2] for i in batch))
        if not has_all_emotions or n_subjects_in_batch < self.n_subjects_min:
            return None

        random.shuffle(batch)
        return batch

    def __iter__(self):
        for _ in range(len(self)):
            batch = self._build_batch()
            if batch is not None:
                yield batch

    def __len__(self):
        return max(1, self.n_samples // self.batch_size)
