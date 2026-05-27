#!/usr/bin/env python3
"""
mumtaffect/pickle_generation.py
================================
Enhanced preprocessing for the AFFEC dataset.

Improvements over MuMTAffect original:
  1. Robust JSON-sidecar column resolution (handles AFFEC v2 width-mismatch bug)
  2. EEG per-trial band-power features (delta/theta/alpha/beta/gamma, optional – requires mne)
  3. Cursor trajectory features (velocity, acceleration, path length)
  4. AU velocity / slope features beyond the original peak/slope
  5. GSR slope feature added to shimmer report
  6. Gender field propagated through all records
  7. Graceful per-trial error handling with detailed console output

Usage:
    python mumtaffect/pickle_generation.py --dataset_path data/raw [--max_participants N] [--skip_eeg]
"""

import os, json, gzip, argparse
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FIXED_LENGTH = 400          # target frames for downsampled time-series
CONFIDENCE_THRESHOLD = 0.7  # minimum OpenFace confidence to keep a frame
EEG_BANDS = {               # band-power frequency ranges (Hz)
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
EEG_SR = 256.0              # EEG sampling rate (Hz) — confirmed from JSON sidecar


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _count_tsv_fields(tsv_gz_path: str) -> int:
    """Count tab-separated fields in the first data row of a gzipped TSV."""
    try:
        with gzip.open(tsv_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
            line = f.readline().rstrip("\n")
        return (line.count("\t") + 1) if line else 0
    except Exception:
        return 0


def _read_json_columns(json_path: str) -> List[str]:
    """Read the Columns list from a BIDS-style JSON sidecar."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        cols = data.get("Columns", [])
        return cols if isinstance(cols, list) else []
    except Exception:
        return []


# Shared cross-participant header cache (populated on first call per modality).
_header_cache: Dict[str, Dict[int, List[str]]] = {}


def _resolve_headers(json_path: str, tsv_gz_path: str, cache_key: str) -> Optional[List[str]]:
    """
    Return robust column headers, handling the AFFEC v2 JSON-width mismatch bug.

    Strategy (mirrors AFFECDataLoader logic):
      1. If JSON column count matches TSV field count → use JSON.
      2. Look up a cached schema of the same width → use cache.
      3. Truncate / pad JSON columns as last resort.
    """
    n_fields = _count_tsv_fields(tsv_gz_path)
    if n_fields <= 0:
        return None

    cols = _read_json_columns(json_path)

    # Populate the per-modality cache from all available JSONs on first call.
    if cache_key not in _header_cache:
        _header_cache[cache_key] = {}
        root = Path(json_path).parents[2]  # data/raw
        pattern = f"*_recording-{cache_key}_physio.json"
        for jp in root.glob(f"*/beh/{pattern}"):
            c = _read_json_columns(str(jp))
            if c:
                _header_cache[cache_key].setdefault(len(c), c)

    if len(cols) == n_fields:
        return cols                             # exact match

    cached = _header_cache[cache_key].get(n_fields)
    if cached:
        return cached                           # matched from cache

    # For very short files (e.g. corrupted 3-column videostream TSVs in AFFEC v2),
    # return the largest known schema so pandas can still parse onset and skip bad lines.
    # AU features will be NaN for these runs, which is handled gracefully downstream.
    if n_fields <= 3 and _header_cache.get(cache_key):
        best = _header_cache[cache_key][max(_header_cache[cache_key])]
        return best

    if cols:
        if len(cols) > n_fields:
            return cols[:n_fields]              # truncate
        return cols + [f"_col{i}" for i in range(len(cols), n_fields)]  # pad

    return None


def _read_modality_tsv(tsv_gz_path: str, json_path: str, cache_key: str,
                        needed_cols: Optional[set] = None) -> Optional[pd.DataFrame]:
    """Read a gzipped BIDS TSV using robust header resolution."""
    headers = _resolve_headers(json_path, tsv_gz_path, cache_key)
    if headers is None:
        return None
    try:
        df = pd.read_csv(
            tsv_gz_path, sep="\t", compression="gzip",
            header=None, engine="python", names=headers,
            on_bad_lines="skip",
            usecols=(lambda c: c in needed_cols) if needed_cols else None,
        )
    except Exception:
        return None

    time_col = "onset" if "onset" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if time_col is None:
        return None
    if time_col != "onset":
        df = df.rename(columns={time_col: "onset"})

    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    if df.empty:
        return None
    df["onset"] = df["onset"] - df["onset"].iloc[0]
    return df


def downsample_interpolate(df: pd.DataFrame, target: int) -> pd.DataFrame:
    """Linear interpolation downsample for numerics; nearest-neighbour for categoricals."""
    n = len(df)
    if n == 0:
        return df
    if n == target:
        return df.copy()
    old_idx = np.arange(n)
    new_idx = np.linspace(0, n - 1, target)
    new_int = np.round(new_idx).astype(int)
    data = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            data[col] = np.interp(new_idx, old_idx, df[col].to_numpy(dtype=float, na_value=0))
        else:
            data[col] = df[col].iloc[new_int].values
    return pd.DataFrame(data)


def _safe_stat(arr: np.ndarray, func):
    v = arr[~np.isnan(arr)]
    return float(func(v)) if v.size > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Eye / Pupil processing
# ─────────────────────────────────────────────────────────────────────────────

EYE_DROP_COLS = [
    "FPOGV", "LPOGV", "RPOGV", "BPOGV", "LPV", "RPV",
    "RPCX", "RPCY", "RPD", "RPS", "LPUPILV", "RPUPILV", "LPD", "LPS", "LPUPILD", "RPUPILD",
]
EYE_NEW_HEADERS = [
    "onset", "TIME", "FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID",
    "LPOGX", "LPOGY", "RPOGX", "RPOGY", "BPOGX", "BPOGY",
    "LPCX", "LPCY", "LEYEX", "LEYEY", "LEYEZ", "REYEX", "REYEY", "REYEZ",
    "duration", "trial_type", "flag", "subject", "run", "trial", "local_time", "stim_file",
    "actual_left_size", "actual_right_size", "actual_avg_size",
]


def _downsample_eye(df: pd.DataFrame, target: int) -> pd.DataFrame:
    df = df.drop(columns=[c for c in EYE_DROP_COLS if c in df.columns])
    return downsample_interpolate(df, target)


def compute_eye_features(eye: pd.DataFrame) -> Dict:
    feats: Dict = {}
    if eye.empty:
        return feats

    trial_time = float(eye["onset"].iloc[-1] - eye["onset"].iloc[0])
    if trial_time <= 0:
        trial_time = 1e-6

    # Fixation
    fix = eye[eye["FPOGV"] == 1].copy() if "FPOGV" in eye.columns else pd.DataFrame()
    if not fix.empty:
        feats["num_fixations"] = fix["FPOGID"].nunique() / trial_time
        feats["fixation_duration_mean"] = float(fix["FPOGD"].mean())
        feats["fixation_duration_std"] = float(fix["FPOGD"].std())
        disps = []
        for _, g in fix.groupby("FPOGID"):
            disps.append(np.sqrt(g["FPOGX"].std() ** 2 + g["FPOGY"].std() ** 2))
        feats["fixation_dispersion_mean"] = float(np.nanmean(disps)) if disps else np.nan
        feats["fixation_dispersion_std"]  = float(np.nanstd(disps))  if disps else np.nan
    else:
        for k in ("num_fixations", "fixation_duration_mean", "fixation_duration_std",
                  "fixation_dispersion_mean", "fixation_dispersion_std"):
            feats[k] = np.nan

    # Pupil sizes
    eye = eye.copy()
    if {"LPD", "LPS", "RPD", "RPS", "RPUPILV"}.issubset(eye.columns):
        eye["actual_left_size"]  = eye["LPD"] * eye["LPS"]
        eye["actual_right_size"] = eye["RPD"] * eye["RPS"]
        eye["actual_avg_size"]   = np.where(
            eye["RPUPILV"] == 1,
            (eye["actual_left_size"] + eye["actual_right_size"]) / 2,
            np.nan,
        )
        ps = eye["actual_avg_size"].dropna().values
        feats["pupil_size_mean"] = _safe_stat(ps, np.mean)
        feats["pupil_size_std"]  = _safe_stat(ps, np.std)
        feats["pupil_size_min"]  = _safe_stat(ps, np.min)
        feats["pupil_size_max"]  = _safe_stat(ps, np.max)
        # Pupil dilation velocity (rate of change)
        if len(ps) > 1:
            feats["pupil_velocity_mean"] = float(np.nanmean(np.abs(np.diff(ps))))
        else:
            feats["pupil_velocity_mean"] = np.nan

        # Blinks (RPUPILV == 0)
        if "RPUPILV" in eye.columns:
            blinks, in_blink, bs = [], False, None
            for _, row in eye.iterrows():
                if row["RPUPILV"] == 0:
                    if not in_blink:
                        in_blink, bs = True, row["onset"]
                else:
                    if in_blink:
                        blinks.append(row["onset"] - bs)
                        in_blink = False
            feats["blink_rate"]          = len(blinks) / trial_time
            feats["blink_duration_mean"] = float(np.mean(blinks)) if blinks else np.nan
            feats["blink_duration_std"]  = float(np.std(blinks))  if blinks else np.nan

    # Saccades (consecutive fixation pairs)
    if not fix.empty:
        fix_list = []
        for fid, g in fix.groupby("FPOGID"):
            fix_list.append({
                "x":     g["BPOGX"].mean() if "BPOGX" in g.columns else g["FPOGX"].mean(),
                "y":     g["BPOGY"].mean() if "BPOGY" in g.columns else g["FPOGY"].mean(),
                "start": float(g["FPOGS"].iloc[0]),
                "end":   float(g["FPOGS"].iloc[0]) + float(g["FPOGD"].iloc[0]),
            })
        fix_list.sort(key=lambda x: x["start"])
        saccades = []
        for i in range(len(fix_list) - 1):
            amp = np.sqrt((fix_list[i+1]["x"] - fix_list[i]["x"]) ** 2 +
                          (fix_list[i+1]["y"] - fix_list[i]["y"]) ** 2)
            dur = max(fix_list[i+1]["start"] - fix_list[i]["end"], 1e-6)
            saccades.append({"amp": amp, "vel": amp / dur})
        if saccades:
            feats["saccade_amplitude_mean"] = float(np.mean([s["amp"] for s in saccades]))
            feats["saccade_velocity_mean"]  = float(np.nanmean([s["vel"] for s in saccades]))
            feats["saccade_rate"]           = len(saccades) / trial_time
        else:
            feats["saccade_amplitude_mean"] = np.nan
            feats["saccade_velocity_mean"]  = np.nan
            feats["saccade_rate"]           = 0.0

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Action Unit processing
# ─────────────────────────────────────────────────────────────────────────────

AU_INTENSITY_COLS = [
    "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r",
    "AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r",
]
AU_BINARY_COLS = [
    "AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c",
    "AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c",
]
FINAL_AU_COLUMNS = (
    ["onset", "confidence", "success"] +
    AU_INTENSITY_COLS + AU_BINARY_COLS +
    ["duration", "trial_type", "flag", "subject", "run", "trial", "local_time", "stim_file"]
)


def compute_au_features(trial: pd.DataFrame) -> Dict:
    feats: Dict = {}
    if trial.empty:
        return feats

    icols = [c for c in AU_INTENSITY_COLS if c in trial.columns]
    bcols = [c for c in AU_BINARY_COLS     if c in trial.columns]

    # Guard: if no intensity columns (e.g. corrupted 3-column videostream file), skip AU-specific features.
    if not icols:
        feats["n_frames"] = len(trial)
        feats["au_data_quality"] = 0.0   # flag as missing
        return feats
    feats["au_data_quality"] = 1.0

    # Basic intensity statistics
    for col in icols:
        v = trial[col].dropna().values
        feats[f"{col}_mean"] = _safe_stat(v, np.mean)
        feats[f"{col}_std"]  = _safe_stat(v, np.std)
        feats[f"{col}_min"]  = _safe_stat(v, np.min)
        feats[f"{col}_max"]  = _safe_stat(v, np.max)

    # Binary activation rates
    for col in bcols:
        feats[f"{col}_activation_rate"] = float(trial[col].mean()) if col in trial.columns else np.nan

    # Confidence
    if "confidence" in trial.columns:
        feats["confidence_mean"] = float(trial["confidence"].mean())
    feats["n_frames"] = len(trial)

    # ── Blink metrics (AU45_c) ────────────────────────────────────────────
    if "AU45_c" in trial.columns and len(trial) > 0:
        blink_col  = trial["AU45_c"].values
        onset_col  = trial["onset"].values
        blink_evts, in_blink, b_start = [], False, 0
        for i, val in enumerate(blink_col):
            if val == 1 and not in_blink:
                in_blink, b_start = True, i
            elif val == 0 and in_blink:
                in_blink = False
                blink_evts.append(onset_col[i] - onset_col[b_start])
        total_dur = max(onset_col[-1] - onset_col[0], 1e-6)
        feats["blink_count"]         = len(blink_evts)
        feats["blink_rate_au"]       = len(blink_evts) / total_dur
        feats["blink_duration_mean_au"] = float(np.mean(blink_evts)) if blink_evts else 0.0

    # ── Dynamic features (peak counts, max slope, mean velocity) ──────────
    if len(trial) >= 2 and "onset" in trial.columns:
        onset = trial["onset"].values
        dt    = np.diff(onset)
        dt[dt == 0] = 1e-6
        for col in icols:
            series = trial[col].values
            peaks = sum(1 for i in range(1, len(series)-1)
                        if series[i] > series[i-1] and series[i] > series[i+1])
            slopes = (series[1:] - series[:-1]) / dt
            feats[f"{col}_peak_count"] = peaks
            feats[f"{col}_max_slope"]  = float(np.max(slopes)) if len(slopes) else 0.0
            # NEW: mean absolute velocity
            feats[f"{col}_mean_velocity"] = float(np.mean(np.abs(slopes)))

    # ── Intensity correlations ─────────────────────────────────────────────
    if len(icols) >= 2:
        data = trial[icols].dropna()
        valid = [c for c in icols if data[c].std() > 1e-6]
        if len(valid) >= 2:
            mat = np.corrcoef(data[valid].values, rowvar=False)
            mat = np.nan_to_num(mat)
            idx = np.triu_indices_from(mat, k=1)
            feats["mean_intensity_corr"] = float(np.mean(mat[idx]))
        else:
            feats["mean_intensity_corr"] = 0.0
    else:
        feats["mean_intensity_corr"] = 0.0

    # ── Composite expressions ─────────────────────────────────────────────
    n = len(trial)
    safe = lambda c: trial[c].values if c in trial.columns else np.zeros(n)
    feats["smile_rate"] = float(np.mean((safe("AU06_c") == 1) & (safe("AU12_c") == 1)))
    feats["frown_rate"] = float(np.mean((safe("AU04_c") == 1) & (safe("AU15_c") == 1)))

    # ── K-Means cluster proportions (6 clusters) ─────────────────────────
    X = trial[icols].dropna().values
    n_clusters = 6
    if X.shape[0] >= n_clusters:
        labels   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
        props    = np.bincount(labels, minlength=n_clusters) / float(len(labels))
        for i, p in enumerate(props):
            feats[f"au_cluster_{i}_prop"] = float(p)
        feats["au_dominant_cluster"] = int(np.argmax(props))
    else:
        for i in range(n_clusters):
            feats[f"au_cluster_{i}_prop"] = 0.0
        feats["au_dominant_cluster"] = -1

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Shimmer (GSR / temperature / accelerometer)
# ─────────────────────────────────────────────────────────────────────────────

FINAL_SHIMMER_COLUMNS = [
    "onset", "Timestamp_raw", "Timestamp_cal", "System_Timestamp_cal",
    "Low_Noise_Accelerometer_X_cal", "Low_Noise_Accelerometer_Y_cal", "Low_Noise_Accelerometer_Z_cal",
    "Wide_Range_Accelerometer_X_cal", "Wide_Range_Accelerometer_Y_cal", "Wide_Range_Accelerometer_Z_cal",
    "Gyroscope_X_cal", "Gyroscope_Y_cal", "Gyroscope_Z_cal",
    "Magnetometer_X_cal", "Magnetometer_Y_cal", "Magnetometer_Z_cal",
    "VSenseBatt_cal", "External_ADC_A7_cal", "Internal_ADC_A13_cal",
    "Pressure_cal", "Temperature_cal", "GSR_raw", "GSR_cal", "GSR_Conductance_cal",
    "duration", "trial_type", "flag", "subject", "run", "trial", "local_time", "stim_file",
]


def _scr_stats(vals: np.ndarray, prefix: str) -> Dict:
    v = vals[~np.isnan(vals)]
    if v.size == 0:
        return {f"{prefix}_{s}": np.nan for s in ("mean", "median", "min", "max", "std")}
    return {
        f"{prefix}_mean":   float(np.mean(v)),
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_min":    float(np.min(v)),
        f"{prefix}_max":    float(np.max(v)),
        f"{prefix}_std":    float(np.std(v)),
    }


def compute_shimmer_features(trial: pd.DataFrame) -> Dict:
    """Extract GSR (via neurokit2 SCR decomposition) + temperature + accelerometer features."""
    feats: Dict = {}
    if trial.empty:
        return feats

    # ── GSR ──────────────────────────────────────────────────────────────
    cond_col = next(
        (c for c in ("GSR_Conductance_cal", "GSR_cal", "GSR_raw") if c in trial.columns), None
    )
    if cond_col is not None:
        raw = pd.to_numeric(trial[cond_col], errors="coerce").dropna().values
        if raw.size >= 4:
            trial_dur = max(trial["onset"].iloc[-1] - trial["onset"].iloc[0], 1.0)
            sr = max(int(len(raw) / trial_dur), 1)
            try:
                std_signal = nk.standardize(raw)
                eda_sig, info = nk.eda_process(std_signal, sampling_rate=sr, method="neurokit")
                onset_mask = eda_sig.get("SCR_Onsets", pd.Series(dtype=float)).fillna(0).astype(bool)
                feats["gsr_scr_n_peaks"] = int(onset_mask.sum())
                for col in ("SCR_Amplitude", "SCR_Height", "SCR_RiseTime", "SCR_Recovery", "SCR_RecoveryTime"):
                    vals = eda_sig[col][onset_mask].to_numpy(dtype=float) if col in eda_sig.columns else np.array([])
                    feats.update(_scr_stats(vals, f"gsr_{col}"))
            except Exception:
                feats["gsr_scr_n_peaks"] = 0
            # NEW: GSR slope over trial
            feats["gsr_slope"] = float(np.polyfit(np.arange(len(raw)), raw, 1)[0]) if len(raw) >= 2 else 0.0

    # ── Temperature ───────────────────────────────────────────────────────
    if "Temperature_cal" in trial.columns:
        temp = pd.to_numeric(trial["Temperature_cal"], errors="coerce").dropna().values
        for stat, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
            feats[f"temp_{stat}"] = _safe_stat(temp, fn)

    # ── Accelerometer magnitude ───────────────────────────────────────────
    acc_cols = ("Low_Noise_Accelerometer_X_cal", "Low_Noise_Accelerometer_Y_cal", "Low_Noise_Accelerometer_Z_cal")
    if all(c in trial.columns for c in acc_cols):
        mag = np.sqrt(sum(pd.to_numeric(trial[c], errors="coerce").fillna(0).values ** 2 for c in acc_cols))
        for stat, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
            feats[f"acc_mag_{stat}"] = _safe_stat(mag, fn)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Cursor processing (NEW)
# ─────────────────────────────────────────────────────────────────────────────

CURSOR_COLS = {"onset", "CX", "CY", "CS"}


def compute_cursor_features(trial: pd.DataFrame) -> Dict:
    """Cursor-trajectory features: path length, velocity, acceleration, convex-hull area."""
    feats: Dict = {}
    if trial.empty or not {"CX", "CY", "onset"}.issubset(trial.columns):
        return feats

    cx = pd.to_numeric(trial["CX"], errors="coerce").fillna(0.0).values
    cy = pd.to_numeric(trial["CY"], errors="coerce").fillna(0.0).values
    t  = pd.to_numeric(trial["onset"], errors="coerce").fillna(0.0).values

    if len(cx) < 2:
        return feats

    # Path length (cumulative Euclidean distance)
    diffs = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    feats["cursor_path_length"] = float(diffs.sum())

    # Velocity (pixels/s)
    dt = np.diff(t)
    dt[dt == 0] = 1e-6
    vel = diffs / dt
    feats["cursor_velocity_mean"] = float(np.mean(vel))
    feats["cursor_velocity_std"]  = float(np.std(vel))
    feats["cursor_velocity_max"]  = float(np.max(vel))

    # Acceleration
    if len(vel) > 1:
        acc = np.abs(np.diff(vel))
        feats["cursor_acceleration_mean"] = float(np.mean(acc))
    else:
        feats["cursor_acceleration_mean"] = 0.0

    # Spatial spread
    feats["cursor_x_std"] = float(np.std(cx))
    feats["cursor_y_std"] = float(np.std(cy))

    # Click count (CS state change → 1)
    if "CS" in trial.columns:
        cs = pd.to_numeric(trial["CS"], errors="coerce").fillna(0).values.astype(int)
        clicks = int(np.sum((cs[1:] == 1) & (cs[:-1] == 0)))
        feats["cursor_click_count"] = clicks

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# EEG processing (NEW — requires mne)
# ─────────────────────────────────────────────────────────────────────────────

def compute_eeg_features(edf_path: str, events: pd.DataFrame,
                          labels: pd.DataFrame) -> List[Dict]:
    """
    Extract per-trial EEG band-power features.

    For each stimulus trial, computes Welch PSD over the video window and
    returns band power for each of 63 channels × 5 bands = 315 features.

    Returns a list of dicts with 'stim_file' and 'eeg_<ch>_<band>' keys.
    """
    try:
        import mne
        from scipy.signal import welch as sp_welch
    except ImportError:
        print("  ⚠  mne not installed — skipping EEG features. pip install mne scipy")
        return []

    results = []
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        ch_names = raw.ch_names
        data_arr, _ = raw[:, :]  # (n_ch, n_times)
    except Exception as e:
        print(f"  ⚠  EEG read error ({edf_path}): {e}")
        return []

    video_evts = events[events.get("flag", pd.Series()) == "video"] if "flag" in events.columns else events
    for _, evt in video_evts.iterrows():
        stim_file = evt.get("stim_file")
        if stim_file is None or pd.isna(stim_file):
            continue
        label_line = labels[labels["stim_file"] == stim_file]
        if label_line.empty:
            continue

        onset    = float(evt["onset"])
        duration = float(evt.get("duration", 3.0))
        start_s  = max(0, int(onset    * sfreq))
        end_s    = min(data_arr.shape[1], int((onset + duration) * sfreq))
        if end_s <= start_s:
            continue

        seg = data_arr[:, start_s:end_s]  # (n_ch, window_samples)
        feat: Dict = {"stim_file": stim_file}
        for ci, ch in enumerate(ch_names):
            ch_seg = seg[ci]
            nperseg = min(int(sfreq), len(ch_seg))
            if nperseg < 4:
                for band in EEG_BANDS:
                    feat[f"eeg_{ch}_{band}"] = np.nan
                continue
            freqs, psd = sp_welch(ch_seg, fs=sfreq, nperseg=nperseg)
            for band, (flo, fhi) in EEG_BANDS.items():
                mask = (freqs >= flo) & (freqs <= fhi)
                feat[f"eeg_{ch}_{band}"] = float(np.mean(psd[mask])) if mask.sum() > 0 else np.nan
        results.append(feat)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-modality processing functions
# ─────────────────────────────────────────────────────────────────────────────

TRIAL_FLAGS = {"trial", "first_fix", "scenario", "second_fix", "video", "last_frame_video"}


def _get_participant_meta(participants_df: pd.DataFrame, user: str) -> Dict:
    row = participants_df[participants_df["participant_id"] == user]
    if row.empty:
        return {}
    r = row.iloc[0]
    # Normalise age column name (dataset has '"age "' with trailing space/quotes)
    age_val = np.nan
    for col in r.index:
        if str(col).strip().strip('"').lower().startswith("age"):
            age_val = pd.to_numeric(r[col], errors="coerce")
            break
    return {
        "openness":          float(r.get("O", np.nan)),
        "conscientiousness": float(r.get("C", np.nan)),
        "extraversion":      float(r.get("E", np.nan)),
        "agreeableness":     float(r.get("A", np.nan)),
        "neuroticism":       float(r.get("N", np.nan)),
        "gender":            str(r.get("gender", "")).strip().lower(),
        "age":               float(age_val) if not pd.isna(age_val) else np.nan,
    }


def process_eye_data(root_path: str, max_participants: Optional[int] = None) -> pd.DataFrame:
    records = []
    ptdf = pd.read_csv(os.path.join(root_path, "participants.tsv"), sep="\t")
    users = list(ptdf["participant_id"].unique())
    if max_participants:
        users = users[:max_participants]

    eye_needed  = {"onset","TIME","FPOGX","FPOGY","FPOGS","FPOGD","FPOGID","FPOGV",
                   "LPOGX","LPOGY","LPOGV","RPOGX","RPOGY","RPOGV","BPOGX","BPOGY","BPOGV"}
    pupil_needed = {"onset","LPCX","LPCY","LPD","LPS","LPV","RPCX","RPCY","RPD","RPS","RPV",
                    "LEYEX","LEYEY","LEYEZ","LPUPILD","LPUPILV","REYEX","REYEY","REYEZ","RPUPILD","RPUPILV"}

    for user in users:
        meta = _get_participant_meta(ptdf, user)
        for run in range(4):
            print(f"  👁  eye  {user} run {run}")
            beh = os.path.join(root_path, user, "beh")
            def gp(stem):
                t = os.path.join(beh, f"{user}_task-fer_run-{run}_recording-{stem}_physio.tsv.gz")
                j = os.path.join(beh, f"{user}_task-fer_run-{run}_recording-{stem}_physio.json")
                return (t, j) if os.path.exists(t) and os.path.exists(j) else (None, None)

            gaze_tsv,  gaze_json  = gp("gaze")
            pupil_tsv, pupil_json = gp("pupil")
            events_path = os.path.join(root_path, user, f"{user}_task-fer_run-{run}_events.tsv")
            labels_path = os.path.join(beh, f"{user}_task-fer_run-{run}_beh.tsv")

            if not all(os.path.exists(p) for p in (events_path, labels_path) if p):
                continue

            gaze_df  = _read_modality_tsv(gaze_tsv, gaze_json, "gaze", eye_needed)   if gaze_tsv  else None
            pupil_df = _read_modality_tsv(pupil_tsv, pupil_json, "pupil", pupil_needed) if pupil_tsv else None

            if gaze_df is None and pupil_df is None:
                continue

            if gaze_df is not None and pupil_df is not None:
                pupil_df = pupil_df.drop(columns=["TIME"], errors="ignore")
                eye_merged = pd.merge_asof(gaze_df.sort_values("onset"),
                                           pupil_df.sort_values("onset"),
                                           on="onset", direction="backward")
            else:
                eye_merged = gaze_df if gaze_df is not None else pupil_df

            # Compute pupil sizes
            if {"LPD","LPS","RPD","RPS","RPUPILV"}.issubset(eye_merged.columns):
                eye_merged["actual_left_size"]  = eye_merged["LPD"] * eye_merged["LPS"]
                eye_merged["actual_right_size"] = eye_merged["RPD"] * eye_merged["RPS"]
                eye_merged["actual_avg_size"]   = np.where(
                    eye_merged["RPUPILV"] == 1,
                    (eye_merged["actual_left_size"] + eye_merged["actual_right_size"]) / 2,
                    np.nan,
                )

            events = pd.read_csv(events_path, sep="\t")
            events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
            events = events.dropna(subset=["onset"]).sort_values("onset")
            labels = pd.read_csv(labels_path, sep="\t")

            eye_merged = pd.merge_asof(eye_merged.sort_values("onset"),
                                       events.sort_values("onset"),
                                       on="onset", direction="backward")

            for stim in labels["stim_file"].dropna().unique():
                try:
                    trial = eye_merged[eye_merged.get("stim_file", pd.Series()) == stim].copy() if "stim_file" in eye_merged.columns else pd.DataFrame()
                    if trial.empty:
                        continue
                    if "flag" in trial.columns:
                        trial = trial[trial["flag"].isin(TRIAL_FLAGS)]
                    if trial.empty:
                        continue

                    eye_feats = compute_eye_features(trial)
                    eye_down  = _downsample_eye(trial, FIXED_LENGTH)

                    lab = labels[labels["stim_file"] == stim].iloc[0]
                    rec = {
                        "user": user, "run": run, "stim_file": stim,
                        "trial": lab["trial"], "stim_emo": lab["trial_type"],
                        "preceived_arousal": lab["p_emotion_a"],
                        "preceived_valance": lab["p_emotion_v"],
                        "felt_arousal": lab["f_emotion_a"],
                        "felt_valance": lab["f_emotion_v"],
                        "Eye_Data_features": eye_feats,
                        "Eye_Data": eye_down,
                        **meta,
                    }
                    records.append(rec)
                except Exception as e:
                    print(f"    ✗ eye {user} run {run} {stim}: {e}")
    return pd.DataFrame(records)


def process_au_data(root_path: str, max_participants: Optional[int] = None) -> pd.DataFrame:
    records = []
    ptdf = pd.read_csv(os.path.join(root_path, "participants.tsv"), sep="\t")
    users = list(ptdf["participant_id"].unique())
    if max_participants:
        users = users[:max_participants]

    for user in users:
        meta = _get_participant_meta(ptdf, user)
        for run in range(4):
            print(f"  🎭  AU   {user} run {run}")
            beh = os.path.join(root_path, user, "beh")
            au_tsv  = os.path.join(beh, f"{user}_task-fer_run-{run}_recording-videostream_physio.tsv.gz")
            au_json = os.path.join(beh, f"{user}_task-fer_run-{run}_recording-videostream_physio.json")
            events_path = os.path.join(root_path, user, f"{user}_task-fer_run-{run}_events.tsv")
            labels_path = os.path.join(beh, f"{user}_task-fer_run-{run}_beh.tsv")

            if not all(os.path.exists(p) for p in (au_tsv, au_json, events_path, labels_path)):
                continue

            au_df = _read_modality_tsv(au_tsv, au_json, "videostream")
            if au_df is None:
                continue

            # Keep numeric AU columns + metadata
            for col in [c for c in AU_INTENSITY_COLS + AU_BINARY_COLS + ["confidence"] if c in au_df.columns]:
                au_df[col] = pd.to_numeric(au_df[col], errors="coerce")

            events = pd.read_csv(events_path, sep="\t")
            events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
            events = events.dropna(subset=["onset"]).sort_values("onset")
            labels = pd.read_csv(labels_path, sep="\t")

            try:
                au_merged = pd.merge_asof(au_df.sort_values("onset"),
                                          events.sort_values("onset"),
                                          on="onset", direction="backward")
            except Exception as e:
                print(f"    ✗ AU merge {user} run {run}: {e}")
                continue

            for stim in labels["stim_file"].dropna().unique():
                try:
                    trial = au_merged[au_merged.get("stim_file", pd.Series()) == stim].copy() if "stim_file" in au_merged.columns else pd.DataFrame()
                    if trial.empty:
                        continue
                    if "flag" in trial.columns:
                        trial = trial[trial["flag"].isin(TRIAL_FLAGS)]
                    if trial.empty:
                        continue

                    au_feats = compute_au_features(trial)
                    au_down  = downsample_interpolate(trial, FIXED_LENGTH)

                    lab = labels[labels["stim_file"] == stim].iloc[0]
                    rec = {
                        "user": user, "run": run, "stim_file": stim,
                        "trial": lab["trial"], "stim_emo": lab["trial_type"],
                        "preceived_arousal": lab["p_emotion_a"],
                        "preceived_valance": lab["p_emotion_v"],
                        "felt_arousal": lab["f_emotion_a"],
                        "felt_valance": lab["f_emotion_v"],
                        "AUs_features": au_feats,
                        "AUs": au_down,
                        **meta,
                    }
                    records.append(rec)
                except Exception as e:
                    print(f"    ✗ AU {user} run {run} {stim}: {e}")
    return pd.DataFrame(records)


def process_shimmer_data(root_path: str, max_participants: Optional[int] = None) -> pd.DataFrame:
    records = []
    ptdf = pd.read_csv(os.path.join(root_path, "participants.tsv"), sep="\t")
    users = list(ptdf["participant_id"].unique())
    if max_participants:
        users = users[:max_participants]

    for user in users:
        meta = _get_participant_meta(ptdf, user)
        for run in range(4):
            print(f"  📡  GSR  {user} run {run}")
            beh = os.path.join(root_path, user, "beh")
            gsr_tsv  = os.path.join(beh, f"{user}_task-fer_run-{run}_recording-gsr_physio.tsv.gz")
            gsr_json = os.path.join(beh, f"{user}_task-fer_run-{run}_recording-gsr_physio.json")
            events_path = os.path.join(root_path, user, f"{user}_task-fer_run-{run}_events.tsv")
            labels_path = os.path.join(beh, f"{user}_task-fer_run-{run}_beh.tsv")

            if not all(os.path.exists(p) for p in (gsr_tsv, gsr_json, events_path, labels_path)):
                continue

            gsr_df = _read_modality_tsv(gsr_tsv, gsr_json, "gsr")
            if gsr_df is None:
                continue

            for col in ("Temperature_cal", "GSR_Conductance_cal", "GSR_cal", "GSR_raw",
                        "Low_Noise_Accelerometer_X_cal", "Low_Noise_Accelerometer_Y_cal",
                        "Low_Noise_Accelerometer_Z_cal"):
                if col in gsr_df.columns:
                    gsr_df[col] = pd.to_numeric(gsr_df[col], errors="coerce")

            events = pd.read_csv(events_path, sep="\t")
            events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
            events = events.dropna(subset=["onset"]).sort_values("onset")
            labels = pd.read_csv(labels_path, sep="\t")

            try:
                gsr_merged = pd.merge_asof(gsr_df.sort_values("onset"),
                                           events.sort_values("onset"),
                                           on="onset", direction="backward")
            except Exception as e:
                print(f"    ✗ shimmer merge {user} run {run}: {e}")
                continue

            for stim in labels["stim_file"].dropna().unique():
                try:
                    trial = gsr_merged[gsr_merged.get("stim_file", pd.Series()) == stim].copy() if "stim_file" in gsr_merged.columns else pd.DataFrame()
                    if trial.empty:
                        continue
                    if "flag" in trial.columns:
                        trial = trial[trial["flag"].isin(TRIAL_FLAGS)]
                    if trial.empty:
                        continue

                    shim_feats = compute_shimmer_features(trial)
                    shim_down  = downsample_interpolate(trial, FIXED_LENGTH)

                    lab = labels[labels["stim_file"] == stim].iloc[0]
                    rec = {
                        "user": user, "run": run, "stim_file": stim,
                        "trial": lab["trial"], "stim_emo": lab["trial_type"],
                        "preceived_arousal": lab["p_emotion_a"],
                        "preceived_valance": lab["p_emotion_v"],
                        "felt_arousal": lab["f_emotion_a"],
                        "felt_valance": lab["f_emotion_v"],
                        "Shimmer_features": shim_feats,
                        "Shimmer": shim_down,
                        **meta,
                    }
                    records.append(rec)
                except Exception as e:
                    print(f"    ✗ shimmer {user} run {run} {stim}: {e}")
    return pd.DataFrame(records)


def process_cursor_data(root_path: str, max_participants: Optional[int] = None) -> pd.DataFrame:
    """NEW: cursor trajectory feature extraction."""
    records = []
    ptdf = pd.read_csv(os.path.join(root_path, "participants.tsv"), sep="\t")
    users = list(ptdf["participant_id"].unique())
    if max_participants:
        users = users[:max_participants]

    for user in users:
        for run in range(4):
            beh = os.path.join(root_path, user, "beh")
            c_tsv  = os.path.join(beh, f"{user}_task-fer_run-{run}_recording-cursor_physio.tsv.gz")
            c_json = os.path.join(beh, f"{user}_task-fer_run-{run}_recording-cursor_physio.json")
            events_path = os.path.join(root_path, user, f"{user}_task-fer_run-{run}_events.tsv")
            labels_path = os.path.join(beh, f"{user}_task-fer_run-{run}_beh.tsv")

            if not all(os.path.exists(p) for p in (c_tsv, c_json, events_path, labels_path)):
                continue

            cur_df = _read_modality_tsv(c_tsv, c_json, "cursor", CURSOR_COLS)
            if cur_df is None:
                continue

            events = pd.read_csv(events_path, sep="\t")
            events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
            events = events.dropna(subset=["onset"]).sort_values("onset")
            labels = pd.read_csv(labels_path, sep="\t")

            try:
                cur_merged = pd.merge_asof(cur_df.sort_values("onset"),
                                           events.sort_values("onset"),
                                           on="onset", direction="backward")
            except Exception:
                continue

            for stim in labels["stim_file"].dropna().unique():
                try:
                    trial = cur_merged[cur_merged.get("stim_file", pd.Series()) == stim].copy() if "stim_file" in cur_merged.columns else pd.DataFrame()
                    if trial.empty:
                        continue
                    if "flag" in trial.columns:
                        trial = trial[trial["flag"].isin(TRIAL_FLAGS)]
                    if trial.empty:
                        continue

                    cur_feats = compute_cursor_features(trial)
                    cur_down  = downsample_interpolate(trial, FIXED_LENGTH)

                    rec = {
                        "user": user, "run": run, "stim_file": stim,
                        "Cursor_features": cur_feats,
                        "Cursor": cur_down,
                    }
                    records.append(rec)
                except Exception:
                    pass

    return pd.DataFrame(records)


def process_eeg_all(root_path: str, max_participants: Optional[int] = None) -> pd.DataFrame:
    """NEW: per-trial EEG band-power features."""
    records = []
    ptdf = pd.read_csv(os.path.join(root_path, "participants.tsv"), sep="\t")
    users = list(ptdf["participant_id"].unique())
    if max_participants:
        users = users[:max_participants]

    for user in users:
        for run in range(4):
            print(f"  🧠  EEG  {user} run {run}")
            edf_path    = os.path.join(root_path, user, "eeg", f"{user}_task-fer_run-{run}_eeg.edf")
            events_path = os.path.join(root_path, user, f"{user}_task-fer_run-{run}_events.tsv")
            labels_path = os.path.join(root_path, user, "beh", f"{user}_task-fer_run-{run}_beh.tsv")

            if not all(os.path.exists(p) for p in (edf_path, events_path, labels_path)):
                continue

            events = pd.read_csv(events_path, sep="\t")
            events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
            events = events.dropna(subset=["onset"])
            labels = pd.read_csv(labels_path, sep="\t")

            trial_feats = compute_eeg_features(edf_path, events, labels)
            for feat in trial_feats:
                feat["user"] = user
                feat["run"]  = run
                records.append(feat)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Merge and save
# ─────────────────────────────────────────────────────────────────────────────

MERGE_KEYS = [
    "user", "run", "stim_file", "trial", "stim_emo",
    "preceived_arousal", "preceived_valance", "felt_arousal", "felt_valance",
]
META_COLS = ["openness", "conscientiousness", "extraversion", "agreeableness",
             "neuroticism", "gender", "age"]


def merge_data(eye_df: pd.DataFrame, au_df: pd.DataFrame,
               shimmer_df: pd.DataFrame,
               cursor_df: Optional[pd.DataFrame] = None,
               eeg_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge all modality dataframes into one trial-level dataframe."""

    # Drop personality/meta from eye and AU before merge to avoid duplicate columns.
    eye_pure    = eye_df.drop(columns=META_COLS, errors="ignore")
    au_pure     = au_df.drop(columns=META_COLS, errors="ignore")
    shim_pure   = shimmer_df.drop(columns=META_COLS, errors="ignore")

    # Shimmer may have different trial/run columns after merge, use looser key set.
    shim_keys = [k for k in MERGE_KEYS if k in shim_pure.columns]

    merged = (
        eye_pure
        .merge(au_pure,   on=MERGE_KEYS, how="inner")
        .merge(shim_pure, on=shim_keys,  how="inner")
    )

    # Add personality / meta from shimmer (which kept them)
    for col in META_COLS:
        if col in shimmer_df.columns and col not in merged.columns:
            lkp = shimmer_df[["user", "run", "stim_file"] + [col]].drop_duplicates()
            merged = merged.merge(lkp, on=["user", "run", "stim_file"], how="left")

    # Cursor (optional)
    if cursor_df is not None and not cursor_df.empty:
        # drop_duplicates can't hash dict/DataFrame cells → deduplicate on key columns only
        cur_pure = (cursor_df[["user", "run", "stim_file", "Cursor_features", "Cursor"]]
                    .groupby(["user", "run", "stim_file"], as_index=False).first())
        merged = merged.merge(cur_pure, on=["user", "run", "stim_file"], how="left")

    # EEG (optional)
    if eeg_df is not None and not eeg_df.empty:
        eeg_pure = eeg_df.drop_duplicates(subset=["user", "run", "stim_file"])
        merged = merged.merge(eeg_pure, on=["user", "run", "stim_file"], how="left")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate AFFEC dataset pickle (enhanced).")
    parser.add_argument("--dataset_path",    type=str, required=True,
                        help="Path to data/raw/ (containing participants.tsv and sub-* folders)")
    parser.add_argument("--output",          type=str, default=None,
                        help="Output pickle path (default: <dataset_path>/dataset_enhanced.pkl)")
    parser.add_argument("--max_participants", type=int, default=None,
                        help="Limit to first N participants (useful for quick tests)")
    parser.add_argument("--skip_eeg",        action="store_true",
                        help="Skip EEG feature extraction (saves time if mne not installed)")
    parser.add_argument("--skip_cursor",     action="store_true",
                        help="Skip cursor feature extraction")
    args = parser.parse_args()

    root = args.dataset_path
    out  = args.output or os.path.join(root, "dataset_enhanced.pkl")
    mp   = args.max_participants

    print("═" * 60)
    print("AFFEC Enhanced Pickle Generation")
    print("═" * 60)

    print("\n[1/5] Processing eye / pupil data …")
    eye_df = process_eye_data(root, mp)
    print(f"      → {len(eye_df)} trials")

    print("\n[2/5] Processing Action Unit data …")
    au_df = process_au_data(root, mp)
    print(f"      → {len(au_df)} trials")

    print("\n[3/5] Processing Shimmer (GSR) data …")
    shim_df = process_shimmer_data(root, mp)
    print(f"      → {len(shim_df)} trials")

    cursor_df = None
    if not args.skip_cursor:
        print("\n[4/5] Processing cursor data …")
        cursor_df = process_cursor_data(root, mp)
        print(f"      → {len(cursor_df)} trials")
    else:
        print("\n[4/5] Cursor processing skipped.")

    eeg_df = None
    if not args.skip_eeg:
        print("\n[5/5] Processing EEG data (band power) …")
        eeg_df = process_eeg_all(root, mp)
        print(f"      → {len(eeg_df)} trials")
    else:
        print("\n[5/5] EEG processing skipped (use --skip_eeg to suppress this).")

    print("\nMerging modalities …")
    df = merge_data(eye_df, au_df, shim_df, cursor_df, eeg_df)
    print(f"Merged dataset: {len(df)} trials × {len(df.columns)} columns")

    df.to_pickle(out)
    print(f"\n✓ Saved to: {out}")

    # Quick summary
    for col in ("user", "stim_emo", "felt_arousal", "preceived_arousal"):
        if col in df.columns:
            print(f"  {col}: {df[col].nunique()} unique values")


if __name__ == "__main__":
    main()
