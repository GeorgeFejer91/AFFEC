"""
PyTorch dataset for MUMT-v2 full-run self-supervised pretraining shards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:
        pass


class MUMTFoundationDataset(Dataset):
    """
    Lazy shard loader for `foundation_preprocess_v2.py` outputs.

    Returns:
        - `x`: normalized sequence tensor `(T, F)`;
        - `mask`: observed-value mask `(T, F)`;
        - metadata fields for user/run/shard.
    """

    def __init__(
        self,
        data_dir: str,
        shard_indices: Optional[Sequence[int]] = None,
        normalize: bool = True,
    ):
        if torch is None:
            raise ImportError("MUMTFoundationDataset requires PyTorch. Install torch to use this dataset.")

        self.data_dir = Path(data_dir)
        self.manifest = pd.read_csv(self.data_dir / "manifest.csv")
        if shard_indices is not None:
            self.manifest = self.manifest.iloc[list(shard_indices)].reset_index(drop=True)
        with open(self.data_dir / "schema.json", "r", encoding="utf-8") as handle:
            self.schema = json.load(handle)

        self.normalize = normalize
        normalization = np.load(self.data_dir / "normalization.npz")
        self.mean = normalization["mean"].astype(np.float32)
        self.std = np.maximum(normalization["std"].astype(np.float32), 1e-6)

        lengths = self.manifest["n_chunks"].astype(int).to_numpy()
        self.ends = np.cumsum(lengths)
        self.starts = self.ends - lengths
        self._cache_shard: Optional[str] = None
        self._cache_x: Optional[np.ndarray] = None
        self._cache_mask: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.ends[-1]) if len(self.ends) else 0

    def _load_shard(self, shard_name: str) -> tuple[np.ndarray, np.ndarray]:
        if shard_name != self._cache_shard:
            data = np.load(self.data_dir / shard_name, allow_pickle=False)
            self._cache_x = data["x"].astype(np.float32)
            self._cache_mask = data["mask"].astype(bool)
            self._cache_shard = shard_name
        return self._cache_x, self._cache_mask

    def __getitem__(self, idx: int) -> dict:
        shard_pos = int(np.searchsorted(self.ends, idx, side="right"))
        local_idx = int(idx - self.starts[shard_pos])
        row = self.manifest.iloc[shard_pos]
        x, mask = self._load_shard(str(row["shard"]))
        sample_x = x[local_idx].copy()
        sample_mask = mask[local_idx].copy()
        if self.normalize:
            sample_x = (sample_x - self.mean) / self.std
            sample_x[~sample_mask] = 0.0

        return {
            "x": torch.tensor(sample_x, dtype=torch.float32),
            "mask": torch.tensor(sample_mask, dtype=torch.bool),
            "user": str(row["user"]),
            "run": int(row["run"]),
            "shard": str(row["shard"]),
        }


def collate_foundation(batch: list[dict]) -> dict:
    if torch is None:
        raise ImportError("collate_foundation requires PyTorch. Install torch to use DataLoader integration.")
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "user": [item["user"] for item in batch],
        "run": [item["run"] for item in batch],
        "shard": [item["shard"] for item in batch],
    }
