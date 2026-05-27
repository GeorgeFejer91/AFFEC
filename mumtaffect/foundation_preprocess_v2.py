"""
Full-run self-supervised pretraining shards for MUMT-v2.

This pipeline is intentionally separate from `preprocess_v2.py`. The supervised
artifact remains one row per labeled trial. This module instead uses the whole
run timeline, aligns modalities onto an auditable 50 ms grid by default, and
saves masks so the model can learn from missing and different-rate modalities
without pretending every signal has the same native quality.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mumtaffect.event_features import AU_INTENSITY_COLS, EEG_REGIONS, EVENT_FLAGS, load_eeg_raw
from mumtaffect.preprocess_v2 import (
    AU_NEEDED,
    CURSOR_NEEDED,
    GAZE_NEEDED,
    GSR_NEEDED,
    PUPIL_NEEDED,
    _eeg_path,
    _events_path,
    _read_events,
    _read_recording,
)


FOUNDATION_EVENT_FLAGS = ("trial", *EVENT_FLAGS)

SEQUENCE_COLUMNS: "OrderedDict[str, Tuple[str, Tuple[str, ...], set]]" = OrderedDict(
    [
        ("eye", ("gaze", ("FPOGX", "FPOGY", "BPOGX", "BPOGY", "FPOGD", "FPOGV"), GAZE_NEEDED)),
        (
            "pupil",
            (
                "pupil",
                ("LPD", "LPS", "RPD", "RPS", "LPUPILD", "RPUPILD", "LPV", "RPV", "LPUPILV", "RPUPILV"),
                PUPIL_NEEDED,
            ),
        ),
        ("au", ("videostream", ("confidence", "success", *AU_INTENSITY_COLS), AU_NEEDED)),
        (
            "gsr",
            (
                "gsr",
                ("GSR_Conductance_cal", "GSR_cal", "GSR_raw", "EDA_Tonic", "EDA_Phasic", "Temperature_cal", "Pressure_cal"),
                GSR_NEEDED,
            ),
        ),
        ("cursor", ("cursor", ("CX", "CY", "CS"), CURSOR_NEEDED)),
    ]
)

EEG_SEQUENCE_FEATURES = tuple(
    f"eeg__{region}__{stat}"
    for region in EEG_REGIONS
    for stat in ("mean", "std")
)


def build_foundation_schema(include_eeg: bool = False) -> Dict[str, object]:
    modality_features: "OrderedDict[str, List[str]]" = OrderedDict()
    for modality, (_, columns, _) in SEQUENCE_COLUMNS.items():
        modality_features[modality] = [f"{modality}__{column}" for column in columns]
    modality_features["event"] = [f"event__{flag}" for flag in FOUNDATION_EVENT_FLAGS]
    if include_eeg:
        modality_features["eeg"] = list(EEG_SEQUENCE_FEATURES)

    feature_names: List[str] = []
    modality_slices: Dict[str, List[int]] = {}
    for modality, names in modality_features.items():
        start = len(feature_names)
        feature_names.extend(names)
        modality_slices[modality] = [start, len(feature_names)]

    return {
        "feature_names": feature_names,
        "modality_slices": modality_slices,
        "include_eeg": include_eeg,
        "event_flags": list(FOUNDATION_EVENT_FLAGS),
    }


def _safe_onsets(df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    if df is None or df.empty or "onset" not in df.columns:
        return None
    values = pd.to_numeric(df["onset"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values if values.size else None


def _run_duration(events: pd.DataFrame, recordings: Mapping[str, Optional[pd.DataFrame]], eeg_raw=None) -> float:
    durations: List[float] = []
    if events is not None and not events.empty:
        onset = pd.to_numeric(events["onset"], errors="coerce")
        duration = pd.to_numeric(events.get("duration", pd.Series(0.0, index=events.index)), errors="coerce").fillna(0.0)
        offsets = (onset + duration).to_numpy(dtype=float)
        offsets = offsets[np.isfinite(offsets)]
        if offsets.size:
            durations.append(float(np.max(offsets)))

    for df in recordings.values():
        onsets = _safe_onsets(df)
        if onsets is not None:
            durations.append(float(np.max(onsets)))

    if eeg_raw is not None:
        try:
            durations.append(float(eeg_raw.times[-1]))
        except Exception:
            pass

    return max(durations) if durations else 0.0


def _unique_sorted(times: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    unique_times, inverse = np.unique(times, return_inverse=True)
    if unique_times.size == times.size:
        return times, values
    sums = np.zeros(unique_times.size, dtype=float)
    counts = np.zeros(unique_times.size, dtype=float)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1.0)
    return unique_times, sums / np.maximum(counts, 1.0)


def _resample_numeric(df: Optional[pd.DataFrame], columns: Sequence[str], grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(grid), len(columns)), dtype=np.float32)
    mask = np.zeros((len(grid), len(columns)), dtype=bool)
    if df is None or df.empty or "onset" not in df.columns:
        return values, mask

    times_all = pd.to_numeric(df["onset"], errors="coerce").to_numpy(dtype=float)
    for column_index, column in enumerate(columns):
        if column not in df.columns:
            continue
        column_values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(times_all) & np.isfinite(column_values)
        if valid.sum() < 2:
            continue

        times, column_values = _unique_sorted(times_all[valid], column_values[valid])
        if times.size < 2:
            continue
        in_range = (grid >= times[0]) & (grid <= times[-1])
        if not np.any(in_range):
            continue
        values[in_range, column_index] = np.interp(grid[in_range], times, column_values).astype(np.float32)
        mask[in_range, column_index] = True
    return values, mask


def _event_features(events: pd.DataFrame, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(grid), len(FOUNDATION_EVENT_FLAGS)), dtype=np.float32)
    if events is None or events.empty or "flag" not in events.columns:
        return values, np.ones_like(values, dtype=bool)

    onset = pd.to_numeric(events["onset"], errors="coerce")
    duration = pd.to_numeric(events.get("duration", pd.Series(0.0, index=events.index)), errors="coerce").fillna(0.0)
    clean_events = events.assign(_onset=onset, _duration=duration).dropna(subset=["_onset"])
    for flag_index, flag in enumerate(FOUNDATION_EVENT_FLAGS):
        active = np.zeros(len(grid), dtype=bool)
        for _, row in clean_events[clean_events["flag"] == flag].iterrows():
            start = float(row["_onset"])
            stop = start + max(float(row["_duration"]), 0.0)
            active |= (grid >= start) & (grid < stop)
        values[:, flag_index] = active.astype(np.float32)
    return values, np.ones_like(values, dtype=bool)


def _normalize_channel_name(channel: str) -> str:
    name = str(channel).upper().replace("EEG", "").replace("-", "").replace("_", "").strip()
    return "".join(ch for ch in name if ch.isalnum())


def _region_indices(channel_names: Sequence[str], max_channels: Optional[int] = None) -> Dict[str, List[int]]:
    usable_names = list(channel_names[:max_channels]) if max_channels else list(channel_names)
    regions: Dict[str, List[int]] = {region: [] for region in EEG_REGIONS}
    for channel_index, channel in enumerate(usable_names):
        clean = _normalize_channel_name(channel)
        for region, prefixes in EEG_REGIONS.items():
            if any(clean.startswith(prefix.upper()) for prefix in prefixes):
                regions[region].append(channel_index)
                break
    return regions


def _eeg_region_features(eeg_raw, grid: np.ndarray, max_channels: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(grid), len(EEG_SEQUENCE_FEATURES)), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    if eeg_raw is None:
        return values, mask

    try:
        channel_names = list(eeg_raw.ch_names[:max_channels]) if max_channels else list(eeg_raw.ch_names)
        picks = [eeg_raw.ch_names.index(channel) for channel in channel_names]
        data = eeg_raw.get_data(picks=picks)
        times = np.arange(data.shape[1], dtype=float) / float(eeg_raw.info["sfreq"])
    except Exception:
        return values, mask

    if data.size == 0 or times.size < 2:
        return values, mask

    regions = _region_indices(channel_names, max_channels=None)
    feature_index = 0
    for region in EEG_REGIONS:
        indices = regions.get(region, [])
        if not indices:
            feature_index += 2
            continue

        region_data = data[indices, :]
        region_mean = np.nanmean(region_data, axis=0)
        region_std = np.nanstd(region_data, axis=0)
        in_range = (grid >= times[0]) & (grid <= times[-1])
        if np.any(in_range):
            values[in_range, feature_index] = np.interp(grid[in_range], times, region_mean).astype(np.float32)
            values[in_range, feature_index + 1] = np.interp(grid[in_range], times, region_std).astype(np.float32)
            mask[in_range, feature_index:feature_index + 2] = True
        feature_index += 2
    return values, mask


def _chunk_run(
    values: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride_len: int,
    max_chunks: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    starts = list(range(0, max(values.shape[0] - seq_len + 1, 0), stride_len))
    if not starts and values.shape[0] >= seq_len:
        starts = [0]
    if max_chunks is not None:
        starts = starts[:max_chunks]
    if not starts:
        return (
            np.zeros((0, seq_len, values.shape[1]), dtype=np.float32),
            np.zeros((0, seq_len, mask.shape[1]), dtype=bool),
        )
    return np.stack([values[start:start + seq_len] for start in starts]), np.stack([mask[start:start + seq_len] for start in starts])


def process_run_to_chunks(
    root: Path,
    user: str,
    run: int,
    schema: Mapping[str, object],
    grid_hz: float,
    chunk_seconds: float,
    stride_seconds: float,
    include_eeg: bool = False,
    eeg_max_channels: Optional[int] = None,
    max_chunks_per_run: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    events_path = _events_path(root, user, run)
    if not events_path.exists():
        return None

    events = _read_events(events_path)
    recordings: Dict[str, Optional[pd.DataFrame]] = {}
    for modality, (recording, _, needed_columns) in SEQUENCE_COLUMNS.items():
        recordings[modality] = _read_recording(root, user, run, recording, needed_columns)

    eeg_raw = None
    if include_eeg:
        eeg_file = _eeg_path(root, user, run)
        if eeg_file.exists():
            eeg_raw = load_eeg_raw(str(eeg_file))

    duration = _run_duration(events, recordings, eeg_raw)
    if not np.isfinite(duration) or duration <= 0:
        return None

    step = 1.0 / float(grid_hz)
    grid = np.arange(0.0, duration, step, dtype=float)
    feature_names = schema["feature_names"]
    modality_slices = schema["modality_slices"]
    values = np.zeros((len(grid), len(feature_names)), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)

    for modality, (_, columns, _) in SEQUENCE_COLUMNS.items():
        start, stop = modality_slices[modality]
        modality_values, modality_mask = _resample_numeric(recordings[modality], columns, grid)
        values[:, start:stop] = modality_values
        mask[:, start:stop] = modality_mask

    start, stop = modality_slices["event"]
    event_values, event_mask = _event_features(events, grid)
    values[:, start:stop] = event_values
    mask[:, start:stop] = event_mask

    if include_eeg and "eeg" in modality_slices:
        start, stop = modality_slices["eeg"]
        eeg_values, eeg_mask = _eeg_region_features(eeg_raw, grid, max_channels=eeg_max_channels)
        values[:, start:stop] = eeg_values
        mask[:, start:stop] = eeg_mask

    seq_len = int(round(chunk_seconds * grid_hz))
    stride_len = max(1, int(round(stride_seconds * grid_hz)))
    chunk_values, chunk_mask = _chunk_run(values, mask, seq_len, stride_len, max_chunks=max_chunks_per_run)
    if chunk_values.shape[0] == 0:
        return None

    return {
        "x": chunk_values,
        "mask": chunk_mask,
        "duration_s": duration,
        "n_grid_steps": len(grid),
        "n_chunks": chunk_values.shape[0],
    }


def _save_npz(path: Path, compress: bool, **arrays) -> None:
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def _reset_output_dir(output_dir: Path) -> None:
    for pattern in ("shard_*.npz", "manifest.csv", "schema.json", "normalization.npz"):
        for path in output_dir.glob(pattern):
            path.unlink()


def generate_foundation_shards(
    dataset_path: str,
    output_dir: str,
    max_participants: Optional[int] = None,
    max_runs: Optional[int] = None,
    include_eeg: bool = False,
    eeg_max_channels: Optional[int] = None,
    grid_hz: float = 20.0,
    chunk_seconds: float = 10.0,
    stride_seconds: float = 5.0,
    max_chunks_per_run: Optional[int] = None,
    compress: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    root = Path(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if any(output_path.glob("shard_*.npz")) and not overwrite:
        raise FileExistsError(f"{output_path} already contains shards. Pass --overwrite to replace them.")
    if overwrite:
        _reset_output_dir(output_path)

    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    users = list(participants["participant_id"].astype(str).unique())
    if max_participants is not None:
        users = users[:max_participants]

    schema = build_foundation_schema(include_eeg=include_eeg)
    schema.update(
        {
            "grid_hz": grid_hz,
            "chunk_seconds": chunk_seconds,
            "stride_seconds": stride_seconds,
            "max_chunks_per_run": max_chunks_per_run,
        }
    )

    runs = range(4 if max_runs is None else max_runs)
    manifest_rows: List[Dict[str, object]] = []
    feature_count = len(schema["feature_names"])
    sum_values = np.zeros(feature_count, dtype=np.float64)
    sum_sq_values = np.zeros(feature_count, dtype=np.float64)
    count_values = np.zeros(feature_count, dtype=np.float64)
    shard_id = 0

    for user_index, user in enumerate(users, start=1):
        print(f"[{user_index}/{len(users)}] {user}", flush=True)
        for run in runs:
            result = process_run_to_chunks(
                root=root,
                user=user,
                run=run,
                schema=schema,
                grid_hz=grid_hz,
                chunk_seconds=chunk_seconds,
                stride_seconds=stride_seconds,
                include_eeg=include_eeg,
                eeg_max_channels=eeg_max_channels,
                max_chunks_per_run=max_chunks_per_run,
            )
            if result is None:
                continue

            x = result["x"]
            mask = result["mask"]
            shard_name = f"shard_{shard_id:05d}_{user}_run{run}.npz"
            _save_npz(output_path / shard_name, compress=compress, x=x, mask=mask)

            observed = mask.astype(np.float64)
            sum_values += np.sum(x.astype(np.float64) * observed, axis=(0, 1))
            sum_sq_values += np.sum((x.astype(np.float64) ** 2) * observed, axis=(0, 1))
            count_values += np.sum(observed, axis=(0, 1))

            manifest_rows.append(
                {
                    "shard": shard_name,
                    "user": user,
                    "run": int(run),
                    "n_chunks": int(result["n_chunks"]),
                    "seq_len": int(round(chunk_seconds * grid_hz)),
                    "n_features": feature_count,
                    "duration_s": float(result["duration_s"]),
                    "n_grid_steps": int(result["n_grid_steps"]),
                }
            )
            print(f"  run {run}: {result['n_chunks']} chunks, {result['duration_s']:.1f}s", flush=True)
            shard_id += 1

    if not manifest_rows:
        raise RuntimeError("No foundation pretraining shards were generated.")

    mean = np.divide(sum_values, count_values, out=np.zeros_like(sum_values), where=count_values > 0)
    variance = np.divide(sum_sq_values, count_values, out=np.zeros_like(sum_sq_values), where=count_values > 0) - mean ** 2
    std = np.sqrt(np.maximum(variance, 1e-8))
    std[count_values <= 0] = 1.0

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_path / "manifest.csv", index=False)
    with open(output_path / "schema.json", "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)
    np.savez(output_path / "normalization.npz", mean=mean.astype(np.float32), std=std.astype(np.float32), count=count_values)

    total_chunks = int(manifest["n_chunks"].sum())
    print(f"Saved {len(manifest)} shards / {total_chunks} chunks × {feature_count} features → {output_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full-run MUMT-v2 self-supervised pretraining shards.")
    parser.add_argument("--dataset_path", required=True, help="Path to AFFEC data/raw directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for shard .npz files and schema.")
    parser.add_argument("--max_participants", type=int, default=None, help="Limit participants for smoke tests.")
    parser.add_argument("--max_runs", type=int, default=None, help="Limit runs per participant for smoke tests.")
    parser.add_argument("--include_eeg", action="store_true", help="Add regional raw EEG mean/std features to the 50 ms grid.")
    parser.add_argument("--eeg_max_channels", type=int, default=None, help="Optional channel limit for EEG smoke tests.")
    parser.add_argument("--grid_hz", type=float, default=20.0, help="Shared pretraining grid rate. 20 Hz = 50 ms.")
    parser.add_argument("--chunk_seconds", type=float, default=10.0, help="Sequence chunk length in seconds.")
    parser.add_argument("--stride_seconds", type=float, default=5.0, help="Stride between chunks in seconds.")
    parser.add_argument("--max_chunks_per_run", type=int, default=None, help="Limit chunks per run for smoke tests.")
    parser.add_argument("--no_compress", action="store_true", help="Use uncompressed .npz shards for faster local writes.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing shards/schema in the output directory.")
    args = parser.parse_args()

    generate_foundation_shards(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_participants=args.max_participants,
        max_runs=args.max_runs,
        include_eeg=args.include_eeg,
        eeg_max_channels=args.eeg_max_channels,
        grid_hz=args.grid_hz,
        chunk_seconds=args.chunk_seconds,
        stride_seconds=args.stride_seconds,
        max_chunks_per_run=args.max_chunks_per_run,
        compress=not args.no_compress,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
