#!/usr/bin/env python3
"""
Generate the event-aware MUMT-v2 dataset artifact.

This preprocessor intentionally does not create one shared 400-frame sequence
for all modalities. It reads each run at native timestamps, extracts features
inside AFFEC event windows, and writes one row per trial.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mumtaffect.audio_features import build_audio_feature_cache
from mumtaffect.event_features import (
    AU_BINARY_COLS,
    AU_INTENSITY_COLS,
    EVENT_FLAGS,
    build_event_windows,
    extract_au_event_features,
    extract_cursor_event_features,
    extract_eeg_event_features_from_raw,
    extract_gaze_event_features,
    extract_gsr_event_features,
    extract_pupil_event_features,
    load_eeg_raw,
    nan_to_none,
)
from mumtaffect.profile_features import context_features, label_features, participant_profile_features


GAZE_NEEDED = {
    "onset", "TIME", "FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID", "FPOGV",
    "LPOGX", "LPOGY", "LPOGV", "RPOGX", "RPOGY", "RPOGV", "BPOGX", "BPOGY", "BPOGV",
}

PUPIL_NEEDED = {
    "onset", "TIME", "LPCX", "LPCY", "LPD", "LPS", "LPV", "RPCX", "RPCY",
    "RPD", "RPS", "RPV", "LEYEX", "LEYEY", "LEYEZ", "LPUPILD", "LPUPILV",
    "REYEX", "REYEY", "REYEZ", "RPUPILD", "RPUPILV",
}

AU_NEEDED = {"onset", "confidence", "success", *AU_INTENSITY_COLS, *AU_BINARY_COLS}

GSR_NEEDED = {
    "onset", "Pressure_cal", "Temperature_cal", "GSR_raw", "GSR_cal", "GSR_Conductance_cal",
    "EDA_Tonic", "EDA_Phasic",
    "Low_Noise_Accelerometer_X_cal", "Low_Noise_Accelerometer_Y_cal", "Low_Noise_Accelerometer_Z_cal",
}

CURSOR_NEEDED = {"onset", "TIME", "CX", "CY", "CS"}


def _count_tsv_fields(tsv_gz_path: str) -> int:
    try:
        with gzip.open(tsv_gz_path, "rt", encoding="utf-8", errors="ignore") as handle:
            line = handle.readline().rstrip("\n")
        return (line.count("\t") + 1) if line else 0
    except Exception:
        return 0


def _read_json_columns(json_path: str) -> List[str]:
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        cols = data.get("Columns", [])
        return cols if isinstance(cols, list) else []
    except Exception:
        return []


_HEADER_CACHE: Dict[str, Dict[int, List[str]]] = {}


def _resolve_headers(json_path: str, tsv_gz_path: str, cache_key: str) -> Optional[List[str]]:
    n_fields = _count_tsv_fields(tsv_gz_path)
    if n_fields <= 0:
        return None

    cols = _read_json_columns(json_path)
    if cache_key not in _HEADER_CACHE:
        _HEADER_CACHE[cache_key] = {}
        root = Path(json_path).parents[2]
        pattern = f"*_recording-{cache_key}_physio.json"
        for candidate in root.glob(f"*/beh/{pattern}"):
            candidate_cols = _read_json_columns(str(candidate))
            if candidate_cols:
                _HEADER_CACHE[cache_key].setdefault(len(candidate_cols), candidate_cols)

    if len(cols) == n_fields:
        return cols

    cached = _HEADER_CACHE[cache_key].get(n_fields)
    if cached:
        return cached

    if n_fields <= 3 and _HEADER_CACHE.get(cache_key):
        return _HEADER_CACHE[cache_key][max(_HEADER_CACHE[cache_key])]

    if cols:
        if len(cols) > n_fields:
            return cols[:n_fields]
        return cols + [f"_col{i}" for i in range(len(cols), n_fields)]

    return None


def _read_modality_tsv(
    tsv_gz_path: str,
    json_path: str,
    cache_key: str,
    needed_cols: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    headers = _resolve_headers(json_path, tsv_gz_path, cache_key)
    if headers is None:
        return None

    try:
        df = pd.read_csv(
            tsv_gz_path,
            sep="\t",
            compression="gzip",
            header=None,
            engine="python",
            names=headers,
            on_bad_lines="skip",
            usecols=(lambda col: col in needed_cols) if needed_cols else None,
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


def _paths(root: Path, user: str, run: int, recording: str) -> tuple[Optional[str], Optional[str]]:
    beh = root / user / "beh"
    tsv = beh / f"{user}_task-fer_run-{run}_recording-{recording}_physio.tsv.gz"
    jsn = beh / f"{user}_task-fer_run-{run}_recording-{recording}_physio.json"
    return (str(tsv), str(jsn)) if tsv.exists() and jsn.exists() else (None, None)


def _read_recording(root: Path, user: str, run: int, recording: str, needed_cols: set) -> Optional[pd.DataFrame]:
    tsv, jsn = _paths(root, user, run, recording)
    if not tsv or not jsn:
        return None
    return _read_modality_tsv(tsv, jsn, recording, needed_cols)


def _events_path(root: Path, user: str, run: int) -> Path:
    return root / user / f"{user}_task-fer_run-{run}_events.tsv"


def _labels_path(root: Path, user: str, run: int) -> Path:
    return root / user / "beh" / f"{user}_task-fer_run-{run}_beh.tsv"


def _eeg_path(root: Path, user: str, run: int) -> Path:
    return root / user / "eeg" / f"{user}_task-fer_run-{run}_eeg.edf"


def _read_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, sep="\t")
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
    events = events.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    return events


def _trial_events(events: pd.DataFrame, label_row: pd.Series) -> pd.DataFrame:
    trial = label_row.get("trial")
    stim_file = label_row.get("stim_file")
    rows = events[events["trial"] == trial] if "trial" in events.columns else pd.DataFrame()
    if rows.empty and "stim_file" in events.columns:
        rows = events[events["stim_file"] == stim_file]
    return rows.sort_values("onset").reset_index(drop=True)


def _event_metadata(windows) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for flag in EVENT_FLAGS:
        window = windows.get(flag)
        features[f"event__{flag}__present"] = float(window is not None)
        features[f"event__{flag}__duration_s"] = float(window.duration) if window else np.nan
        features[f"event__{flag}__onset_s"] = float(window.onset) if window else np.nan
    return features


def _base_record(user: str, run: int, label_row: pd.Series, participants: pd.DataFrame) -> Dict:
    stim_emo = label_row.get("trial_type")
    record: Dict = {
        "user": user,
        "run": int(run),
        "trial": int(label_row.get("trial")),
        "stim_file": str(label_row.get("stim_file")),
    }
    record.update(context_features(stim_emo))
    record.update(label_features(label_row))
    record.update(participant_profile_features(participants, user))
    return record


def process_run(
    root: Path,
    user: str,
    run: int,
    participants: pd.DataFrame,
    include_eeg: bool = False,
    audio_cache: Optional[Dict[str, Dict[str, float]]] = None,
    eeg_max_channels: Optional[int] = None,
) -> list[Dict]:
    events_path = _events_path(root, user, run)
    labels_path = _labels_path(root, user, run)
    if not events_path.exists() or not labels_path.exists():
        return []

    events = _read_events(events_path)
    labels = pd.read_csv(labels_path, sep="\t")

    gaze = _read_recording(root, user, run, "gaze", GAZE_NEEDED)
    pupil = _read_recording(root, user, run, "pupil", PUPIL_NEEDED)
    au = _read_recording(root, user, run, "videostream", AU_NEEDED)
    gsr = _read_recording(root, user, run, "gsr", GSR_NEEDED)
    cursor = _read_recording(root, user, run, "cursor", CURSOR_NEEDED)

    eeg_raw = None
    if include_eeg:
        eeg_path = _eeg_path(root, user, run)
        if eeg_path.exists():
            eeg_raw = load_eeg_raw(str(eeg_path))

    rows = []
    for _, label_row in labels.iterrows():
        trial_events = _trial_events(events, label_row)
        windows = build_event_windows(trial_events)
        if not windows:
            continue

        record = _base_record(user, run, label_row, participants)
        record.update(_event_metadata(windows))
        record.update(extract_gaze_event_features(gaze, windows))
        record.update(extract_pupil_event_features(pupil, windows))
        record.update(extract_au_event_features(au, windows))
        record.update(extract_gsr_event_features(gsr, windows))
        record.update(extract_cursor_event_features(cursor, windows))

        if eeg_raw is not None:
            record.update(extract_eeg_event_features_from_raw(eeg_raw, windows, max_channels=eeg_max_channels))

        if audio_cache is not None:
            record.update(audio_cache.get(str(label_row.get("stim_file")), {}))

        rows.append(nan_to_none(record))
    return rows


def generate_dataset(
    dataset_path: str,
    output: Optional[str] = None,
    max_participants: Optional[int] = None,
    include_eeg: bool = False,
    media_root: Optional[str] = None,
    eeg_max_channels: Optional[int] = None,
) -> pd.DataFrame:
    root = Path(dataset_path)
    participants_path = root / "participants.tsv"
    participants = pd.read_csv(participants_path, sep="\t")
    users = list(participants["participant_id"].unique())
    if max_participants is not None:
        users = users[:max_participants]

    audio_cache = None
    if media_root:
        stim_files = []
        for user in users:
            for run in range(4):
                labels_path = _labels_path(root, user, run)
                if labels_path.exists():
                    labels = pd.read_csv(labels_path, sep="\t")
                    stim_files.extend(labels["stim_file"].dropna().astype(str).tolist())
        audio_cache = build_audio_feature_cache(stim_files, media_root)

    all_rows = []
    for user_idx, user in enumerate(users, start=1):
        print(f"[{user_idx}/{len(users)}] {user}")
        for run in range(4):
            rows = process_run(
                root=root,
                user=user,
                run=run,
                participants=participants,
                include_eeg=include_eeg,
                audio_cache=audio_cache,
                eeg_max_channels=eeg_max_channels,
            )
            if rows:
                print(f"  run {run}: {len(rows)} trials")
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            df.to_csv(out_path, index=False)
        else:
            df.to_pickle(out_path)
        print(f"Saved {len(df)} trials × {len(df.columns)} columns → {out_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate event-aware MUMT-v2 dataset.")
    parser.add_argument("--dataset_path", required=True, help="Path to AFFEC data/raw directory.")
    parser.add_argument("--output", default="data/raw/dataset_mumt_v2.pkl", help="Output .pkl or .csv path.")
    parser.add_argument("--max_participants", type=int, default=None, help="Limit participants for smoke tests.")
    parser.add_argument("--include_eeg", action="store_true", help="Extract event-aware EEG features. Slower.")
    parser.add_argument("--eeg_max_channels", type=int, default=None, help="Optional channel limit for EEG smoke tests.")
    parser.add_argument("--media_root", default=None, help="Optional root containing stimulus audio/video media.")
    args = parser.parse_args()

    generate_dataset(
        dataset_path=args.dataset_path,
        output=args.output,
        max_participants=args.max_participants,
        include_eeg=args.include_eeg,
        media_root=args.media_root,
        eeg_max_channels=args.eeg_max_channels,
    )


if __name__ == "__main__":
    main()
