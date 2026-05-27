"""
mumtaffect/dataset.py
=====================
PyTorch Dataset for the enhanced AFFEC pickle.

Handles:
  • Eye, Pupil, AU, GSR sequences (variable length, resampled by model)
  • Cursor sequences (optional)
  • EEG band-power features (315-dim, optional)
  • Trial-level static features (eye, AU, shimmer stats)
  • Emotion binning (9-point scale → 3 bins: 0/1/2)
  • Gender encoding (f→0, m→1)
  • Data augmentation (noise injection)
  • Per-modality StandardScaler normalisation

Performance: all pandas DataFrame → numpy conversions are done once in __init__
via a bulk cache build (_build_cache).  __getitem__ is then pure numpy/PyTorch
array indexing — roughly 100-200× faster than the naive approach.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import resample

# ─────────────────────────────────────────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────────────────────────────────────────

STIM_EMO_CATEGORIES = ["neutral", "sad", "happy", "angry", "fear", "disgust"]

COMMON_FLAG_CATEGORIES = [
    "trial", "first_fix", "scenario", "second_fix",
    "video", "last_frame_video", "f_emotion_labelling", "p_emotion_labelling",
]

GAZE_COLS   = ["onset","FPOGX","FPOGY","FPOGS","FPOGD","FPOGID",
               "LPOGX","LPOGY","RPOGX","RPOGY","BPOGX","BPOGY",
               "LPCX","LPCY","LEYEX","LEYEY","LEYEZ","REYEX","REYEY","REYEZ","flag"]

PUPIL_COLS  = ["onset","actual_left_size","actual_right_size","actual_avg_size","flag"]

AU_COLS     = ["onset","confidence","success",
               "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r",
               "AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r",
               "AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c",
               "AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c",
               "flag"]

GSR_COLS    = ["onset","Pressure_cal","Temperature_cal","GSR_raw","GSR_cal",
               "EDA_Tonic","EDA_Phasic","flag"]

CURSOR_COLS = ["onset","CX","CY","CS","flag"]

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def stim_emo_onehot(stim_emo) -> np.ndarray:
    vec = np.zeros(len(STIM_EMO_CATEGORIES), dtype=np.float32)
    if isinstance(stim_emo, str) and stim_emo in STIM_EMO_CATEGORIES:
        vec[STIM_EMO_CATEGORIES.index(stim_emo)] = 1.0
    return vec


def process_modal_data(modal_df: pd.DataFrame, desired_cols: list,
                        flag_categories: list) -> np.ndarray:
    """
    Extract desired columns from a modal DataFrame,
    one-hot-encode the 'flag' column, fill NaNs → float32 array.
    """
    avail = [c for c in desired_cols if c in modal_df.columns]
    df    = modal_df[avail].copy().fillna(0)

    if "flag" in df.columns:
        dummies = pd.get_dummies(df["flag"], prefix="flag")
        expected = [f"flag_{cat}" for cat in flag_categories]
        dummies = dummies.reindex(columns=expected, fill_value=0)
        df = pd.concat([df.drop(columns=["flag"]), dummies], axis=1)

    return df.values.astype(np.float32)


def _bulk_process_modality(modal_dfs: list, desired_cols: list,
                            flag_categories: list) -> list:
    """
    Process a list of modal DataFrames in bulk (single pd.concat + ops).
    Returns a list of numpy float32 arrays, one per input DataFrame.
    Orders-of-magnitude faster than calling process_modal_data in a loop.
    """
    if not modal_dfs:
        return []

    # Filter cols to those that exist in at least the first df
    sample_df = next((d for d in modal_dfs if d is not None and isinstance(d, pd.DataFrame)), None)
    if sample_df is None:
        return [np.zeros((0,), dtype=np.float32)] * len(modal_dfs)

    avail = [c for c in desired_cols if c in sample_df.columns]

    # Build a row-count index for splitting after concat
    valid_mask = [isinstance(d, pd.DataFrame) and not d.empty for d in modal_dfs]
    lengths    = [len(d) if ok else 0 for d, ok in zip(modal_dfs, valid_mask)]

    # Concatenate all valid DataFrames at once
    valid_dfs = [d[avail].fillna(0) for d, ok in zip(modal_dfs, valid_mask) if ok]
    if not valid_dfs:
        # Return zero arrays matching expected output width
        out_w = (len(avail) - 1) + len(flag_categories)  # -1 for flag col
        return [np.zeros((l, out_w), dtype=np.float32) for l in lengths]

    combined = pd.concat(valid_dfs, ignore_index=True)

    if "flag" in combined.columns:
        dummies  = pd.get_dummies(combined["flag"], prefix="flag")
        expected = [f"flag_{cat}" for cat in flag_categories]
        dummies  = dummies.reindex(columns=expected, fill_value=0)
        combined = pd.concat([combined.drop(columns=["flag"]), dummies], axis=1)

    arr = combined.values.astype(np.float32)
    out_w = arr.shape[1]

    # Split back into per-sample arrays
    results  = []
    pos      = 0
    valid_it = iter(range(len([x for x in valid_mask if x])))
    for ok, l in zip(valid_mask, lengths):
        if ok:
            results.append(arr[pos: pos + l])
            pos += l
        else:
            results.append(np.zeros((0, out_w), dtype=np.float32))

    return results


def flatten_dict(d: dict) -> np.ndarray:
    """Flatten a feature dict to a sorted-key float32 vector."""
    vals = []
    for key in sorted(d.keys()):
        val = d[key]
        if isinstance(val, (list, np.ndarray)):
            vals.append(float(np.nanmean(val)))
        elif isinstance(val, dict):
            vals.extend([float(v) if not isinstance(v, (list, np.ndarray))
                         else float(np.nanmean(v)) for v in val.values()])
        else:
            try:
                vals.append(float(val))
            except (TypeError, ValueError):
                vals.append(0.0)
    arr = np.array(vals, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def flatten_shimmer_features(row) -> np.ndarray:
    """
    Flatten Shimmer_features dict.

    Handles two formats:
      • Flat (enhanced pickle): {"gsr_scr_n_peaks": 4, "gsr_slope": -0.0005, ...}
      • Nested (original MuMTAffect pickle): {"GSR_report": {...}, "Temperature_report": {...}, ...}
    """
    sf = row.get("Shimmer_features", {})
    if not isinstance(sf, dict):
        return np.zeros(0, dtype=np.float32)

    # Detect nested format by looking for known sub-dict keys
    if any(k in sf for k in ("GSR_report", "Temperature_report", "Accelerometer_report")):
        parts = []
        for sub in ("GSR_report", "Temperature_report", "Accelerometer_report"):
            sub_dict = sf.get(sub, {})
            if isinstance(sub_dict, dict):
                parts.extend([float(v) if not isinstance(v, (list, np.ndarray))
                               else float(np.nanmean(v))
                               for v in sub_dict.values()])
        arr = np.array(parts, dtype=np.float32)
    else:
        # Flat format — just call flatten_dict
        arr = flatten_dict(sf)

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def eeg_feature_vector(row, eeg_cols: list) -> np.ndarray:
    """Extract EEG band-power features; return zeros if missing."""
    if not eeg_cols:
        return np.zeros(0, dtype=np.float32)
    vals = np.array([float(row.get(c, 0.0) or 0.0) for c in eeg_cols], dtype=np.float32)
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def time_warp(signal: np.ndarray, factor_range=(0.9, 1.1)) -> np.ndarray:
    """Randomly speed-up or slow-down a (T, D) sequence."""
    factor   = np.random.uniform(*factor_range)
    new_len  = max(1, int(signal.shape[0] * factor))
    return resample(signal, new_len, axis=0)


def random_crop(signal: np.ndarray, crop_size: int) -> np.ndarray:
    if signal.shape[0] <= crop_size:
        return signal
    start = np.random.randint(0, signal.shape[0] - crop_size)
    return signal[start: start + crop_size]


def noise_injection(signal: np.ndarray, std=0.01) -> np.ndarray:
    return signal + np.random.normal(0, std, signal.shape).astype(np.float32)


def mixup_samples(s1: tuple, s2: tuple, lam: float) -> tuple:
    """Mix two dataset samples with coefficient lam."""
    out = []
    for i, (a, b) in enumerate(zip(s1, s2)):
        if i == 7:                          # user_id → sentinel
            out.append(torch.tensor(-1, dtype=a.dtype))
        elif i in (6, 11):                  # emotion_binned, gender → soft mix
            out.append(lam * a.float() + (1 - lam) * b.float())
        elif isinstance(a, torch.Tensor):
            out.append(lam * a + (1 - lam) * b)
        else:
            out.append(lam * a + (1 - lam) * b)
    return tuple(out)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    """
    Returns a 15-element tuple per sample:

        (eye_seq, pupil_seq, au_seq, gsr_seq,
         cursor_seq,                           # zeros if cursor not in pickle
         stim_emo_vec,
         personality,
         emotion_binned,
         user_id,
         eye_features, au_features, shimmer_features,
         gender,
         eeg_features,
         cursor_features)

    All pandas → numpy conversion is done once in __init__ (_build_cache).
    __getitem__ is then pure numpy array indexing + optional scaler.transform
    + optional noise augmentation — roughly 100-200× faster than naive access.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        selected_emotions: list,
        user2idx: dict,
        modality_scalers=None,
        device=None,
        augment: bool = False,
        verbose: bool = True,
    ):
        self.df                = df.reset_index(drop=True)
        self.selected_emotions = selected_emotions
        self.user2idx          = user2idx
        self.modality_scalers  = modality_scalers or {}
        self.device            = device or torch.device("cpu")
        self.augment           = augment
        self.verbose           = verbose

        # Detect EEG columns in the dataframe
        self.eeg_cols = sorted([c for c in df.columns if c.startswith("eeg_")])

        # Detect cursor availability
        self.has_cursor = "Cursor" in df.columns

        # ── Pre-compute expected static feature dims ─────────────────────
        # Scan first valid rows; missing rows use zeros of the same size.
        def _first_valid_feat(col, extractor):
            for _, row in df.iterrows():
                val = row.get(col)
                if isinstance(val, dict) and val:
                    try:
                        return len(extractor(val))
                    except Exception:
                        continue
            return 0

        self._eye_feat_dim    = _first_valid_feat("Eye_Data_features", flatten_dict)
        self._au_feat_dim     = _first_valid_feat("AUs_features",      flatten_dict)
        self._cursor_feat_dim = _first_valid_feat("Cursor_features",   flatten_dict)

        self._shim_feat_dim = 0
        for _, row in df.iterrows():
            v = flatten_shimmer_features(row)
            if len(v) > 0:
                self._shim_feat_dim = len(v)
                break

        # ── Build the numpy cache (one-time cost) ─────────────────────────
        self._build_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Cache construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_cache(self):
        """
        Convert every row's DataFrame columns to numpy arrays exactly once,
        using bulk pd.concat operations per modality.  After this, __getitem__
        is pure numpy array indexing.
        """
        df  = self.df
        n   = len(df)
        _cursor_proc_dim = (len(CURSOR_COLS) - 1) + len(COMMON_FLAG_CATEGORIES)

        if self.verbose:
            print(f"    [Dataset] caching {n} rows … ", end="", flush=True)

        # ── Sequence modalities (bulk) ────────────────────────────────────
        eye_modal_dfs    = list(df["Eye_Data"])
        au_modal_dfs     = list(df["AUs"])
        gsr_modal_dfs    = list(df["Shimmer"])

        self._eye_seqs   = _bulk_process_modality(eye_modal_dfs,  GAZE_COLS,   COMMON_FLAG_CATEGORIES)
        self._pupil_seqs = _bulk_process_modality(eye_modal_dfs,  PUPIL_COLS,  COMMON_FLAG_CATEGORIES)
        self._au_seqs    = _bulk_process_modality(au_modal_dfs,   AU_COLS,     COMMON_FLAG_CATEGORIES)
        self._gsr_seqs   = _bulk_process_modality(gsr_modal_dfs,  GSR_COLS,    COMMON_FLAG_CATEGORIES)

        if self.has_cursor:
            cursor_modal_dfs = list(df["Cursor"])
            raw_cursor = _bulk_process_modality(cursor_modal_dfs, CURSOR_COLS, COMMON_FLAG_CATEGORIES)
            self._cursor_seqs = []
            for i, arr in enumerate(raw_cursor):
                if arr.shape[0] > 0:
                    self._cursor_seqs.append(arr)
                else:
                    # Fallback: zeros matching the eye sequence length
                    t = self._eye_seqs[i].shape[0] if self._eye_seqs[i].shape[0] > 0 else 400
                    self._cursor_seqs.append(np.zeros((t, _cursor_proc_dim), dtype=np.float32))
        else:
            self._cursor_seqs = []
            for i in range(n):
                t = self._eye_seqs[i].shape[0] if self._eye_seqs[i].shape[0] > 0 else 400
                self._cursor_seqs.append(np.zeros((t, _cursor_proc_dim), dtype=np.float32))

        # ── Static feature arrays (pre-allocated, filled row by row) ──────
        stim_emo_vecs   = np.zeros((n, len(STIM_EMO_CATEGORIES)),    dtype=np.float32)
        personalities   = np.zeros((n, 5),                            dtype=np.float32)
        n_emo           = len(self.selected_emotions)
        emotion_binneds = np.zeros((n, n_emo),                        dtype=np.int64)
        user_ids        = np.zeros(n,                                  dtype=np.int64)
        gender_ints     = np.zeros(n,                                  dtype=np.int64)
        eye_feats_arr   = np.zeros((n, max(self._eye_feat_dim,  1)),   dtype=np.float32)
        au_feats_arr    = np.zeros((n, max(self._au_feat_dim,   1)),   dtype=np.float32)
        shim_feats_arr  = np.zeros((n, max(self._shim_feat_dim, 1)),   dtype=np.float32)
        eeg_feats_arr   = np.zeros((n, max(len(self.eeg_cols),  1)),   dtype=np.float32)
        cursor_feats_arr= np.zeros((n, max(self._cursor_feat_dim, 1)), dtype=np.float32)

        def _safe_feat(raw, extractor, expected_dim):
            if isinstance(raw, dict) and raw:
                try:
                    v = extractor(raw)
                    if len(v) == expected_dim:
                        return v
                except Exception:
                    pass
            return np.zeros(expected_dim, dtype=np.float32)

        for i, (_, row) in enumerate(df.iterrows()):
            stim_emo_vecs[i] = stim_emo_onehot(row.get("stim_emo"))
            personalities[i] = [float(row.get(c, 0) or 0)
                                 for c in ("openness","conscientiousness","extraversion",
                                           "agreeableness","neuroticism")]
            emotion_raw        = np.array([row[e] for e in self.selected_emotions], dtype=np.float64)
            emotion_binneds[i] = np.clip((emotion_raw.astype(int) - 1) // 3, 0, 2)
            user_ids[i]        = self.user2idx.get(row["user"], 0)
            gv                 = str(row.get("gender", "m")).strip().lower()
            gender_ints[i]     = 0 if gv.startswith("f") else 1

            if self._eye_feat_dim > 0:
                eye_feats_arr[i]   = _safe_feat(row.get("Eye_Data_features"), flatten_dict, self._eye_feat_dim)
            if self._au_feat_dim > 0:
                au_feats_arr[i]    = _safe_feat(row.get("AUs_features"),      flatten_dict, self._au_feat_dim)

            sf = flatten_shimmer_features(row)
            if self._shim_feat_dim > 0 and len(sf) == self._shim_feat_dim:
                shim_feats_arr[i]  = sf

            if self.has_cursor and self._cursor_feat_dim > 0:
                cursor_feats_arr[i] = _safe_feat(row.get("Cursor_features"), flatten_dict, self._cursor_feat_dim)

            if self.eeg_cols:
                ev = eeg_feature_vector(row, self.eeg_cols)
                eeg_feats_arr[i, :len(ev)] = ev

        # Apply personality z-score scaler if provided (fitted on training split).
        # Raw BFI scores span [9, 49]; normalising to ~N(0,1) lets the model
        # head use unbounded linear outputs and makes the loss well-conditioned.
        sc_pers = self.modality_scalers.get("personality")
        if sc_pers is not None:
            personalities = sc_pers.transform(personalities).astype(np.float32)

        self._stim_emo_vecs    = stim_emo_vecs
        self._personalities    = personalities
        self._emotion_binneds  = emotion_binneds
        self._user_ids         = user_ids
        self._gender_ints      = gender_ints
        self._eye_feats_arr    = eye_feats_arr
        self._au_feats_arr     = au_feats_arr
        self._shim_feats_arr   = shim_feats_arr
        self._eeg_feats_arr    = eeg_feats_arr
        self._cursor_feats_arr = cursor_feats_arr

        if self.verbose:
            print("done.", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset interface
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.df)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def __getitem__(self, idx):
        # ── Retrieve pre-cached sequences ────────────────────────────────
        if self.augment:
            # Copy so noise injection doesn't corrupt the cache
            eye_seq    = self._eye_seqs[idx].copy()
            pupil_seq  = self._pupil_seqs[idx].copy()
            au_seq     = self._au_seqs[idx].copy()
            gsr_seq    = self._gsr_seqs[idx].copy()
            cursor_seq = self._cursor_seqs[idx].copy()
        else:
            eye_seq    = self._eye_seqs[idx]
            pupil_seq  = self._pupil_seqs[idx]
            au_seq     = self._au_seqs[idx]
            gsr_seq    = self._gsr_seqs[idx]
            cursor_seq = self._cursor_seqs[idx]

        # ── Optional per-modality normalisation ───────────────────────────
        def _scale(arr: np.ndarray, key: str) -> np.ndarray:
            scaler = self.modality_scalers.get(key)
            if scaler is None:
                return arr
            try:
                return scaler.transform(arr)
            except ValueError:
                return arr

        eye_seq    = _scale(eye_seq,    "Eye_Data")
        au_seq     = _scale(au_seq,     "AUs")
        gsr_seq    = _scale(gsr_seq,    "Shimmer")
        cursor_seq = _scale(cursor_seq, "Cursor")

        # ── Augmentation (train mode only) ────────────────────────────────
        if self.augment:
            for seq in (eye_seq, pupil_seq, au_seq, gsr_seq, cursor_seq):
                if np.random.rand() < 0.3:
                    seq[:] = noise_injection(seq)

        # ── Return (pure numpy / tensor ops from here) ────────────────────
        return (
            self._to_tensor(eye_seq),
            self._to_tensor(pupil_seq),
            self._to_tensor(au_seq),
            self._to_tensor(gsr_seq),
            self._to_tensor(cursor_seq),
            self._to_tensor(self._stim_emo_vecs[idx]),
            self._to_tensor(self._personalities[idx]),
            torch.tensor(self._emotion_binneds[idx], dtype=torch.long,  device=self.device),
            torch.tensor(self._user_ids[idx],        dtype=torch.long,  device=self.device),
            self._to_tensor(self._eye_feats_arr[idx]),
            self._to_tensor(self._au_feats_arr[idx]),
            self._to_tensor(self._shim_feats_arr[idx]),
            torch.tensor(self._gender_ints[idx],     dtype=torch.long,  device=self.device),
            self._to_tensor(self._eeg_feats_arr[idx]),
            self._to_tensor(self._cursor_feats_arr[idx]),
        )
