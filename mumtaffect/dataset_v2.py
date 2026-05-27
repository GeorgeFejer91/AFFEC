"""
PyTorch dataset utilities for MUMT-v2 event-aware artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:
        pass


RAW_TARGETS = (
    "felt_arousal_raw",
    "felt_valence_raw",
    "perceived_arousal_raw",
    "perceived_valence_raw",
)

BIN_TARGETS = (
    "felt_arousal_bin",
    "felt_valence_bin",
    "perceived_arousal_bin",
    "perceived_valence_bin",
)

GAP_TARGETS = (
    "gap_arousal_raw",
    "gap_valence_raw",
)

DEFAULT_FEATURE_PREFIXES = (
    "eye__",
    "pupil__",
    "au__",
    "gsr__",
    "cursor__",
    "eeg__",
    "audio__",
    "profile__",
    "context__",
    "event__",
    "quality__",
    "labelstyle__",
    "foundation__",
)

NON_FEATURE_COLUMNS = {
    "user",
    "run",
    "trial",
    "stim_file",
    "stim_emo",
    *RAW_TARGETS,
    *BIN_TARGETS,
    *GAP_TARGETS,
    "gap_arousal_abs",
    "gap_valence_abs",
    "label__trial_index",
    "label__run_index",
    "label_rt__felt_arousal",
    "label_rt__felt_valence",
    "label_rt__perceived_arousal",
    "label_rt__perceived_valence",
}


@dataclass
class MUMTV2FeatureSpec:
    feature_columns: list[str]
    raw_target_columns: tuple[str, ...] = RAW_TARGETS
    bin_target_columns: tuple[str, ...] = BIN_TARGETS
    gap_target_columns: tuple[str, ...] = GAP_TARGETS


def select_feature_columns(
    df: pd.DataFrame,
    include_prefixes: Sequence[str] = DEFAULT_FEATURE_PREFIXES,
    exclude_prefixes: Sequence[str] = (),
) -> list[str]:
    columns = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if exclude_prefixes and any(col.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if include_prefixes and not any(col.startswith(prefix) for prefix in include_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(col)
    return sorted(columns)


def fit_v2_scaler(df: pd.DataFrame, feature_columns: Sequence[str]) -> StandardScaler:
    scaler = StandardScaler()
    values = df.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    scaler.fit(values)
    return scaler


def split_support_query(
    df: pd.DataFrame,
    k: int,
    user_col: str = "user",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split each user into k support trials and remaining query trials.

    Use after selecting held-out users for few-shot evaluation.
    """
    rng = np.random.default_rng(seed)
    support_idx = []
    query_idx = []
    for _, user_df in df.groupby(user_col):
        indices = user_df.index.to_numpy()
        if len(indices) == 0:
            continue
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_support = min(k, len(shuffled))
        support_idx.extend(shuffled[:n_support].tolist())
        query_idx.extend(shuffled[n_support:].tolist())
    return np.array(support_idx, dtype=int), np.array(query_idx, dtype=int)


class MUMTV2Dataset(Dataset):
    """
    Dataset for `dataset_mumt_v2.pkl`.

    Returns a dictionary to keep the v2 model interface explicit:

    - `features`: all selected numeric features
    - `labels_raw`: four 1--9 emotion ratings
    - `labels_bin`: four low/mid/high targets
    - `labels_gap`: felt-perceived arousal/valence gaps
    - `user_index`: participant index
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[Sequence[str]] = None,
        scaler: Optional[StandardScaler] = None,
        include_prefixes: Sequence[str] = DEFAULT_FEATURE_PREFIXES,
        exclude_prefixes: Sequence[str] = (),
        user2idx: Optional[dict[str, int]] = None,
        device: Optional["torch.device"] = None,
    ):
        if torch is None:
            raise ImportError("MUMTV2Dataset requires PyTorch. Install torch to use the training dataset wrapper.")
        self.df = df.reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns) if feature_columns is not None else select_feature_columns(
            self.df,
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
        )
        self.scaler = scaler
        self.device = device or torch.device("cpu")
        self.user2idx = user2idx or {user: idx for idx, user in enumerate(sorted(self.df["user"].astype(str).unique()))}

        self.spec = MUMTV2FeatureSpec(feature_columns=self.feature_columns)

    def __len__(self) -> int:
        return len(self.df)

    def _features(self, row: pd.Series) -> torch.Tensor:
        values = pd.to_numeric(row.reindex(self.feature_columns), errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        if self.scaler is not None:
            values = self.scaler.transform(values.reshape(1, -1)).reshape(-1).astype(np.float32)
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        labels_raw = pd.to_numeric(row.reindex(RAW_TARGETS), errors="coerce").to_numpy(dtype=np.float32)
        labels_bin = pd.to_numeric(row.reindex(BIN_TARGETS), errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        labels_gap = pd.to_numeric(row.reindex(GAP_TARGETS), errors="coerce").to_numpy(dtype=np.float32)
        user = str(row["user"])
        return {
            "features": self._features(row),
            "labels_raw": torch.tensor(labels_raw, dtype=torch.float32, device=self.device),
            "labels_bin": torch.tensor(labels_bin, dtype=torch.long, device=self.device),
            "labels_gap": torch.tensor(labels_gap, dtype=torch.float32, device=self.device),
            "user_index": torch.tensor(self.user2idx.get(user, 0), dtype=torch.long, device=self.device),
            "user": user,
            "stim_file": str(row.get("stim_file", "")),
        }


def collate_v2(batch: list[dict]) -> dict:
    if torch is None:
        raise ImportError("collate_v2 requires PyTorch. Install torch to use DataLoader integration.")
    tensor_keys = ("features", "labels_raw", "labels_bin", "labels_gap", "user_index")
    out = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    out["user"] = [item["user"] for item in batch]
    out["stim_file"] = [item["stim_file"] for item in batch]
    return out
