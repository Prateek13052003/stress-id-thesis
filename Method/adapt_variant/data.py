"""
Loads the 5-modality StressID feature table (ECG, EDA, RESP, Video, Audio) built
by Feature Extraction/build_multimodal_dataset.py, and wraps it for training a
physio-anchored missing-modality model.

Modality order is fixed as MODALITIES below, with ECG as index 0 (the anchor):
physiology is ~100% available in StressID and is the only modality realistically
collectible outside a lab (wearable sensor, no camera/microphone), unlike ADAPT's
original choice of video as anchor.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEAT_DIR = "Feature Extraction/Features"
DATASET_DIR = "Dataset"

MODALITIES = ["ecg", "eda", "resp", "video", "audio"]
ANCHOR = "ecg"


def load_modality_column_groups():
    X = pd.read_csv(f"{FEAT_DIR}/multimodal_5modality_features.csv", index_col=0)
    groups = {}
    for name in MODALITIES:
        cols = [c for c in X.columns if c.startswith(f"{name}_")]
        groups[name] = cols
    return X, groups


def load_labels(target="binary-stress"):
    labels = pd.read_csv(f"{DATASET_DIR}/labels.csv", index_col=0)
    return labels[target].dropna()


class ModalityStats:
    """Per-modality column mean/std computed from available (non-missing) rows
    of the TRAINING split only, used to standardize + zero-fill missing cells."""

    def __init__(self, X, mask, groups, train_idx):
        self.mean, self.std = {}, {}
        for name, cols in groups.items():
            avail = mask.loc[train_idx, name]
            vals = X.loc[train_idx, cols][avail]
            self.mean[name] = vals.mean().fillna(0.0)
            self.std[name] = vals.std().replace(0, 1.0).fillna(1.0)

    def transform(self, X, mask, groups):
        """Returns a dict[name] -> standardized array with missing rows zeroed."""
        out = {}
        for name, cols in groups.items():
            block = (X[cols] - self.mean[name]) / self.std[name]
            block = block.fillna(0.0)
            avail = mask[name].values.astype(bool)
            block = block.values.copy()
            block[~avail] = 0.0
            out[name] = block.astype(np.float32)
        return out


class StressIDMultimodalDataset(Dataset):
    def __init__(self, X, mask, y, groups, stats):
        self.arrays = stats.transform(X, mask, groups)
        self.mask = mask[list(groups.keys())].values.astype(bool)
        self.y = y.values.astype(np.int64)
        self.index = list(X.index)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        modality_feats = {name: torch.from_numpy(arr[i]) for name, arr in self.arrays.items()}
        avail = torch.from_numpy(self.mask[i])
        return modality_feats, avail, self.y[i]


def build_full_table(target="binary-stress"):
    """Aligns the 5-modality feature table + mask + labels on a common index."""
    X, groups = load_modality_column_groups()
    mask = pd.read_csv(f"{FEAT_DIR}/multimodal_5modality_mask.csv", index_col=0)
    y = load_labels(target)
    idx = X.index.intersection(mask.index).intersection(y.index)
    return X.loc[idx], mask.loc[idx], y.loc[idx], groups


def subject_groups(index):
    return np.array([i.split("_")[0] for i in index])
