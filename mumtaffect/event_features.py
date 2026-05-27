"""
Event-aware feature extraction for MUMT-v2.

The functions in this module avoid forcing all modalities into one common
sequence length. Each modality is summarized inside AFFEC event windows and
emits auditable feature names with modality/event prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


EVENT_FLAGS = (
    "first_fix",
    "scenario",
    "second_fix",
    "video",
    "last_frame_video",
    "f_emotion_labelling",
    "p_emotion_labelling",
)

BASELINE_FLAG = "first_fix"
VIDEO_FLAG = "video"

AU_INTENSITY_COLS = (
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
)

AU_BINARY_COLS = (
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
)

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

EEG_REGIONS = {
    "frontal": ("Fp", "AF", "F"),
    "central": ("FC", "C", "CP"),
    "temporal": ("FT", "T", "TP"),
    "parietal": ("P", "PO"),
    "occipital": ("O",),
}


@dataclass(frozen=True)
class EventWindow:
    flag: str
    onset: float
    offset: float
    duration: float


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_array(values: Iterable) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _sanitize_name(name: object) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "_").replace("-", "_")


def _stats(values: Iterable, prefix: str, features: Dict[str, float]) -> None:
    arr = _safe_array(values)
    features[f"{prefix}__count"] = float(arr.size)
    if arr.size == 0:
        for stat in ("mean", "std", "min", "max", "median", "q25", "q75"):
            features[f"{prefix}__{stat}"] = np.nan
        return

    features[f"{prefix}__mean"] = float(np.mean(arr))
    features[f"{prefix}__std"] = float(np.std(arr))
    features[f"{prefix}__min"] = float(np.min(arr))
    features[f"{prefix}__max"] = float(np.max(arr))
    features[f"{prefix}__median"] = float(np.median(arr))
    features[f"{prefix}__q25"] = float(np.quantile(arr, 0.25))
    features[f"{prefix}__q75"] = float(np.quantile(arr, 0.75))


def _slope(times: Iterable, values: Iterable) -> float:
    t = _safe_array(times)
    v = _safe_array(values)
    if t.size != v.size or t.size < 2 or np.nanstd(t) <= 1e-9:
        return np.nan
    try:
        return float(np.polyfit(t - t[0], v, 1)[0])
    except Exception:
        return np.nan


def _mean_abs_velocity(times: Iterable, values: Iterable) -> float:
    t = _safe_array(times)
    v = _safe_array(values)
    if t.size != v.size or t.size < 2:
        return np.nan
    dt = np.diff(t)
    dv = np.diff(v)
    valid = np.abs(dt) > 1e-9
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs(dv[valid] / dt[valid])))


def build_event_windows(events: pd.DataFrame, flags: Iterable[str] = EVENT_FLAGS) -> Dict[str, EventWindow]:
    windows: Dict[str, EventWindow] = {}
    if events is None or events.empty:
        return windows

    events = events.copy()
    events["onset"] = _to_numeric(events["onset"])
    events["duration"] = _to_numeric(events.get("duration", pd.Series(np.nan, index=events.index)))
    for flag in flags:
        rows = events[events["flag"] == flag] if "flag" in events.columns else pd.DataFrame()
        if rows.empty:
            continue
        row = rows.iloc[0]
        onset = float(row["onset"])
        duration = float(row["duration"]) if np.isfinite(row["duration"]) else 0.0
        windows[flag] = EventWindow(flag=flag, onset=onset, offset=onset + max(duration, 0.0), duration=max(duration, 0.0))
    return windows


def slice_event(df: Optional[pd.DataFrame], window: EventWindow) -> pd.DataFrame:
    if df is None or df.empty or "onset" not in df.columns:
        return pd.DataFrame()
    data = df.copy()
    data["onset"] = _to_numeric(data["onset"])
    return data[(data["onset"] >= window.onset) & (data["onset"] < window.offset)].copy()


def _add_quality(features: Dict[str, float], modality: str, flag: str, data: pd.DataFrame, duration: float) -> None:
    base = f"quality__{modality}__{flag}"
    n_samples = len(data)
    features[f"{base}__n_samples"] = float(n_samples)
    features[f"{base}__duration_s"] = float(duration)
    features[f"{base}__sample_rate_est"] = float(n_samples / duration) if duration > 0 else np.nan
    features[f"{base}__has_data"] = float(n_samples > 0)


def _entropy_2d(x: np.ndarray, y: np.ndarray, bins: int = 4) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    hist, _, _ = np.histogram2d(x[valid], y[valid], bins=bins)
    probs = hist.ravel()
    probs = probs[probs > 0] / np.sum(probs)
    return float(-np.sum(probs * np.log2(probs))) if probs.size else np.nan


def _pupil_sizes(pupil: pd.DataFrame) -> pd.Series:
    if "actual_avg_size" in pupil.columns:
        return _to_numeric(pupil["actual_avg_size"])
    required = {"LPD", "LPS", "RPD", "RPS"}
    if required.issubset(pupil.columns):
        left = _to_numeric(pupil["LPD"]) * _to_numeric(pupil["LPS"])
        right = _to_numeric(pupil["RPD"]) * _to_numeric(pupil["RPS"])
        if "RPUPILV" in pupil.columns:
            valid = _to_numeric(pupil["RPUPILV"]) == 1
            return ((left + right) / 2).where(valid)
        return (left + right) / 2
    return pd.Series(dtype=float)


def extract_gaze_event_features(gaze: Optional[pd.DataFrame], windows: Mapping[str, EventWindow]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for flag, window in windows.items():
        data = slice_event(gaze, window)
        _add_quality(features, "eye", flag, data, window.duration)
        prefix = f"eye__{flag}"
        if data.empty:
            continue

        for col in ("FPOGX", "FPOGY", "BPOGX", "BPOGY", "LPOGX", "LPOGY", "RPOGX", "RPOGY", "FPOGD"):
            if col in data.columns:
                _stats(data[col], f"{prefix}__{col}", features)
                if col not in ("FPOGD",):
                    features[f"{prefix}__{col}__slope"] = _slope(data["onset"], data[col])
                    features[f"{prefix}__{col}__mean_abs_velocity"] = _mean_abs_velocity(data["onset"], data[col])

        valid_cols = [col for col in ("FPOGV", "LPOGV", "RPOGV", "BPOGV") if col in data.columns]
        for col in valid_cols:
            vals = _to_numeric(data[col])
            features[f"{prefix}__{col}__valid_ratio"] = float(np.nanmean(vals == 1)) if len(vals) else np.nan

        if "FPOGID" in data.columns:
            ids = data["FPOGID"].dropna()
            features[f"{prefix}__fixation_count"] = float(ids.nunique())
            features[f"{prefix}__fixation_rate"] = float(ids.nunique() / window.duration) if window.duration > 0 else np.nan

        x_col = "BPOGX" if "BPOGX" in data.columns else ("FPOGX" if "FPOGX" in data.columns else None)
        y_col = "BPOGY" if "BPOGY" in data.columns else ("FPOGY" if "FPOGY" in data.columns else None)
        if x_col and y_col:
            x = _to_numeric(data[x_col]).to_numpy(dtype=float)
            y = _to_numeric(data[y_col]).to_numpy(dtype=float)
            features[f"{prefix}__gaze_entropy_4x4"] = _entropy_2d(x, y, bins=4)
            features[f"{prefix}__gaze_spread"] = float(np.sqrt(np.nanvar(x) + np.nanvar(y))) if len(x) else np.nan
    return features


def extract_pupil_event_features(pupil: Optional[pd.DataFrame], windows: Mapping[str, EventWindow]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for flag, window in windows.items():
        data = slice_event(pupil, window)
        _add_quality(features, "pupil", flag, data, window.duration)
        prefix = f"pupil__{flag}"
        if data.empty:
            continue

        sizes = _pupil_sizes(data)
        if len(sizes):
            _stats(sizes, f"{prefix}__size", features)
            features[f"{prefix}__size__slope"] = _slope(data["onset"], sizes)
            features[f"{prefix}__size__mean_abs_velocity"] = _mean_abs_velocity(data["onset"], sizes)

        for col in ("LPUPILV", "RPUPILV", "LPV", "RPV"):
            if col in data.columns:
                vals = _to_numeric(data[col])
                features[f"{prefix}__{col}__valid_ratio"] = float(np.nanmean(vals == 1)) if len(vals) else np.nan

        if "RPUPILV" in data.columns:
            blink = _to_numeric(data["RPUPILV"]) == 0
            features[f"{prefix}__blink_sample_ratio"] = float(np.nanmean(blink)) if len(blink) else np.nan

    _add_video_baseline_deltas(features, "pupil", "size")
    return features


def extract_au_event_features(au: Optional[pd.DataFrame], windows: Mapping[str, EventWindow], confidence_threshold: float = 0.7) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for flag, window in windows.items():
        data = slice_event(au, window)
        _add_quality(features, "au", flag, data, window.duration)
        prefix = f"au__{flag}"
        if data.empty:
            continue

        if "confidence" in data.columns:
            conf = _to_numeric(data["confidence"])
            _stats(conf, f"{prefix}__confidence", features)
            features[f"{prefix}__high_conf_ratio"] = float(np.nanmean(conf >= confidence_threshold)) if len(conf) else np.nan
            usable = data[conf >= confidence_threshold].copy()
            if usable.empty:
                usable = data
        else:
            usable = data

        if "success" in usable.columns:
            success = _to_numeric(usable["success"])
            features[f"{prefix}__success_ratio"] = float(np.nanmean(success == 1)) if len(success) else np.nan

        for col in AU_INTENSITY_COLS:
            if col in usable.columns:
                _stats(usable[col], f"{prefix}__{col}", features)
                features[f"{prefix}__{col}__slope"] = _slope(usable["onset"], usable[col])
                features[f"{prefix}__{col}__mean_abs_velocity"] = _mean_abs_velocity(usable["onset"], usable[col])

        for col in AU_BINARY_COLS:
            if col in usable.columns:
                vals = _to_numeric(usable[col])
                features[f"{prefix}__{col}__activation_rate"] = float(np.nanmean(vals == 1)) if len(vals) else np.nan

        def binary(name: str) -> np.ndarray:
            if name not in usable.columns:
                return np.zeros(len(usable), dtype=bool)
            return (_to_numeric(usable[name]) == 1).to_numpy(dtype=bool)

        if len(usable):
            features[f"{prefix}__smile_rate"] = float(np.mean(binary("AU06_c") & binary("AU12_c")))
            features[f"{prefix}__frown_rate"] = float(np.mean(binary("AU04_c") & binary("AU15_c")))
            features[f"{prefix}__attention_surprise_rate"] = float(np.mean(binary("AU01_c") | binary("AU02_c") | binary("AU05_c")))
            features[f"{prefix}__mouth_open_rate"] = float(np.mean(binary("AU25_c") | binary("AU26_c")))
    return features


def extract_gsr_event_features(gsr: Optional[pd.DataFrame], windows: Mapping[str, EventWindow]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for flag, window in windows.items():
        data = slice_event(gsr, window)
        _add_quality(features, "gsr", flag, data, window.duration)
        prefix = f"gsr__{flag}"
        if data.empty:
            continue

        conductance_col = next((col for col in ("GSR_Conductance_cal", "GSR_cal", "GSR_raw") if col in data.columns), None)
        if conductance_col:
            vals = _to_numeric(data[conductance_col])
            _stats(vals, f"{prefix}__conductance", features)
            features[f"{prefix}__conductance__slope"] = _slope(data["onset"], vals)
            features[f"{prefix}__conductance__mean_abs_velocity"] = _mean_abs_velocity(data["onset"], vals)

        for col in ("EDA_Tonic", "EDA_Phasic", "Temperature_cal", "Pressure_cal"):
            if col in data.columns:
                _stats(data[col], f"{prefix}__{col}", features)
                features[f"{prefix}__{col}__slope"] = _slope(data["onset"], data[col])

        acc_cols = ("Low_Noise_Accelerometer_X_cal", "Low_Noise_Accelerometer_Y_cal", "Low_Noise_Accelerometer_Z_cal")
        if all(col in data.columns for col in acc_cols):
            mag = np.sqrt(sum(_to_numeric(data[col]).fillna(0).to_numpy(dtype=float) ** 2 for col in acc_cols))
            _stats(mag, f"{prefix}__acc_mag", features)

    _add_video_baseline_deltas(features, "gsr", "conductance")
    _add_video_baseline_deltas(features, "gsr", "EDA_Tonic")
    _add_video_baseline_deltas(features, "gsr", "EDA_Phasic")
    return features


def extract_cursor_event_features(cursor: Optional[pd.DataFrame], windows: Mapping[str, EventWindow]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for flag, window in windows.items():
        data = slice_event(cursor, window)
        _add_quality(features, "cursor", flag, data, window.duration)
        prefix = f"cursor__{flag}"
        if data.empty:
            continue

        for col in ("CX", "CY", "CS"):
            if col in data.columns:
                _stats(data[col], f"{prefix}__{col}", features)

        if {"CX", "CY", "onset"}.issubset(data.columns) and len(data) >= 2:
            x = _to_numeric(data["CX"]).ffill().fillna(0).to_numpy(dtype=float)
            y = _to_numeric(data["CY"]).ffill().fillna(0).to_numpy(dtype=float)
            t = _to_numeric(data["onset"]).to_numpy(dtype=float)
            dt = np.diff(t)
            valid = np.abs(dt) > 1e-9
            dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            features[f"{prefix}__path_length"] = float(np.sum(dist))
            if np.any(valid):
                velocity = dist[valid] / dt[valid]
                _stats(velocity, f"{prefix}__velocity", features)

        if "CS" in data.columns and len(data) >= 2:
            cs = _to_numeric(data["CS"]).fillna(0).to_numpy(dtype=int)
            features[f"{prefix}__click_count"] = float(np.sum((cs[1:] == 1) & (cs[:-1] == 0)))
    return features


def _add_video_baseline_deltas(features: Dict[str, float], modality: str, signal_name: str) -> None:
    base = f"{modality}__{BASELINE_FLAG}__{signal_name}"
    video = f"{modality}__{VIDEO_FLAG}__{signal_name}"
    out = f"{modality}__video_minus_baseline__{signal_name}"
    for stat in ("mean", "median", "min", "max", "std", "q25", "q75"):
        b_key = f"{base}__{stat}"
        v_key = f"{video}__{stat}"
        if b_key in features and v_key in features:
            features[f"{out}__{stat}"] = features[v_key] - features[b_key]


def extract_eeg_event_features_from_raw(raw, windows: Mapping[str, EventWindow], max_channels: Optional[int] = None) -> Dict[str, float]:
    """Extract event-level EEG log-band-power features from an already loaded MNE Raw object."""
    try:
        from scipy.signal import welch
    except ImportError:
        return {}

    features: Dict[str, float] = {}
    sfreq = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names[:max_channels]) if max_channels else list(raw.ch_names)
    picks = [raw.ch_names.index(ch) for ch in channel_names]

    for flag, window in windows.items():
        prefix = f"eeg__{flag}"
        start = max(0, int(window.onset * sfreq))
        stop = max(start, int(window.offset * sfreq))
        data, _ = raw[picks, start:stop]
        features[f"quality__eeg__{flag}__n_samples"] = float(data.shape[1])
        features[f"quality__eeg__{flag}__duration_s"] = float(window.duration)
        features[f"quality__eeg__{flag}__has_data"] = float(data.shape[1] > 0)
        if data.shape[1] < 4:
            continue

        nperseg = min(int(sfreq), data.shape[1])
        freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=1)
        channel_band_values: Dict[str, Dict[str, float]] = {band: {} for band in EEG_BANDS}
        for band, (low, high) in EEG_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            band_values = np.log(np.mean(psd[:, mask], axis=1) + 1e-12) if np.any(mask) else np.full(len(channel_names), np.nan)
            for ch_name, value in zip(channel_names, band_values):
                clean_ch = _sanitize_name(ch_name)
                features[f"{prefix}__{clean_ch}__{band}_log_power"] = float(value)
                channel_band_values[band][ch_name] = float(value)

        for band, values_by_channel in channel_band_values.items():
            all_values = np.array(list(values_by_channel.values()), dtype=float)
            if all_values.size:
                features[f"{prefix}__all_channels__{band}_log_power__mean"] = float(np.nanmean(all_values))
                features[f"{prefix}__all_channels__{band}_log_power__std"] = float(np.nanstd(all_values))
            for region, starts in EEG_REGIONS.items():
                vals = [
                    value for channel, value in values_by_channel.items()
                    if any(str(channel).startswith(start) for start in starts)
                ]
                if vals:
                    features[f"{prefix}__{region}__{band}_log_power__mean"] = float(np.nanmean(vals))

    for band in EEG_BANDS:
        _add_video_baseline_deltas(features, "eeg", f"all_channels__{band}_log_power")
    return features


def load_eeg_raw(edf_path: str, bandpass: tuple[float, float] = (1.0, 45.0), notch_freq: Optional[float] = None):
    """Load and lightly filter EEG. Returns None if MNE or the EDF file is unavailable."""
    try:
        import mne
    except ImportError:
        return None

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        if notch_freq:
            raw.notch_filter(freqs=[notch_freq], verbose=False)
        if bandpass:
            raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)
        return raw
    except Exception:
        return None


def nan_to_none(features: Dict[str, object]) -> Dict[str, object]:
    """Normalize non-finite numeric values while preserving identifiers."""
    clean = {}
    for key, value in features.items():
        if isinstance(value, (str, bool, int, np.integer)):
            clean[key] = value
            continue
        try:
            val = float(value)
        except Exception:
            clean[key] = value
            continue
        clean[key] = val if np.isfinite(val) else np.nan
    return clean
