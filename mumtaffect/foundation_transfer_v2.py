"""
Transfer a pretrained MUMT-v2 foundation encoder into supervised trial features.

The extractor pools hidden states from the full-run foundation model over each
trial's event window (default: video watching) and writes deterministic
`foundation__<event>__dim*` columns into a supervised MUMT-v2 pickle.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from mumtaffect.foundation_model_v2 import MUMTFoundationModel


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_schema(foundation_dir: Path) -> Dict:
    with open(foundation_dir / "schema.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_foundation_model(checkpoint_path: Path, schema: Dict, device: torch.device) -> MUMTFoundationModel:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_schema = checkpoint.get("schema", {})
    if checkpoint_schema.get("feature_names") and checkpoint_schema["feature_names"] != schema["feature_names"]:
        raise ValueError("Checkpoint feature schema does not match the foundation shard schema.")

    args = checkpoint.get("args", {})
    model = MUMTFoundationModel(
        n_features=len(schema["feature_names"]),
        max_seq_len=int(schema["chunk_seconds"] * schema["grid_hz"]),
        d_model=int(args.get("d_model", 128)),
        n_heads=int(args.get("n_heads", 4)),
        n_layers=int(args.get("n_layers", 3)),
        hidden_dim=int(args.get("hidden_dim", 256)),
        dropout=float(args.get("dropout", 0.15)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_normalization(foundation_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    normalization = np.load(foundation_dir / "normalization.npz")
    mean = normalization["mean"].astype(np.float32)
    std = np.maximum(normalization["std"].astype(np.float32), 1e-6)
    return mean, std


def _read_supervised_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_pickle(path)


def _write_supervised_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_pickle(path)


def _window_columns(event_flag: str) -> Tuple[str, str]:
    return f"event__{event_flag}__onset_s", f"event__{event_flag}__duration_s"


def _build_window_index(df: pd.DataFrame, event_flag: str) -> Dict[Tuple[str, int], List[Tuple[int, float, float]]]:
    onset_col, duration_col = _window_columns(event_flag)
    if onset_col not in df.columns or duration_col not in df.columns:
        raise ValueError(f"Missing `{onset_col}` / `{duration_col}` in supervised table.")

    windows: Dict[Tuple[str, int], List[Tuple[int, float, float]]] = defaultdict(list)
    onsets = pd.to_numeric(df[onset_col], errors="coerce")
    durations = pd.to_numeric(df[duration_col], errors="coerce")
    for row_index, row in df.iterrows():
        onset = float(onsets.iloc[row_index]) if np.isfinite(onsets.iloc[row_index]) else np.nan
        duration = float(durations.iloc[row_index]) if np.isfinite(durations.iloc[row_index]) else np.nan
        if not np.isfinite(onset) or not np.isfinite(duration) or duration <= 0:
            continue
        key = (str(row["user"]), int(row["run"]))
        windows[key].append((int(row_index), onset, onset + duration))
    return windows


def _normalize_chunks(x: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_norm = (x.astype(np.float32) - mean) / std
    x_norm[~mask] = 0.0
    return x_norm


def _pool_chunk_hidden(
    hidden: np.ndarray,
    chunk_offset: int,
    run_windows: Sequence[Tuple[int, float, float]],
    grid_hz: float,
    chunk_seconds: float,
    stride_seconds: float,
    embedding_sum: np.ndarray,
    embedding_steps: np.ndarray,
) -> None:
    seq_len = hidden.shape[1]
    for local_chunk_index in range(hidden.shape[0]):
        global_chunk_index = chunk_offset + local_chunk_index
        chunk_start = global_chunk_index * stride_seconds
        chunk_stop = chunk_start + chunk_seconds
        for row_index, window_start, window_stop in run_windows:
            overlap_start = max(window_start, chunk_start)
            overlap_stop = min(window_stop, chunk_stop)
            if overlap_stop <= overlap_start:
                continue

            local_start = max(0, int(np.floor((overlap_start - chunk_start) * grid_hz)))
            local_stop = min(seq_len, int(np.ceil((overlap_stop - chunk_start) * grid_hz)))
            if local_stop <= local_start:
                continue

            segment = hidden[local_chunk_index, local_start:local_stop, :]
            embedding_sum[row_index] += segment.sum(axis=0)
            embedding_steps[row_index] += segment.shape[0]


def add_foundation_embeddings(
    pickle_path: str,
    foundation_dir: str,
    checkpoint: str,
    output: str,
    event_flag: str = "video",
    batch_size: int = 64,
    device: str = "auto",
    max_rows: Optional[int] = None,
    log_interval: int = 20,
) -> pd.DataFrame:
    supervised_path = Path(pickle_path)
    foundation_path = Path(foundation_dir)
    checkpoint_path = Path(checkpoint)
    output_path = Path(output)

    df = _read_supervised_table(supervised_path).reset_index(drop=True)
    if max_rows is not None:
        df = df.iloc[:max_rows].copy().reset_index(drop=True)

    schema = _load_schema(foundation_path)
    mean, std = _load_normalization(foundation_path)
    torch_device = resolve_device(device)
    model = _load_foundation_model(checkpoint_path, schema, torch_device)
    d_model = int(model.position_embedding.shape[-1])

    windows = _build_window_index(df, event_flag)
    manifest = pd.read_csv(foundation_path / "manifest.csv")
    grid_hz = float(schema["grid_hz"])
    chunk_seconds = float(schema["chunk_seconds"])
    stride_seconds = float(schema["stride_seconds"])

    embedding_sum = np.zeros((len(df), d_model), dtype=np.float64)
    embedding_steps = np.zeros(len(df), dtype=np.float64)
    processed_shards = 0

    with torch.no_grad():
        for shard_index, shard_row in manifest.iterrows():
            key = (str(shard_row["user"]), int(shard_row["run"]))
            run_windows = windows.get(key)
            if not run_windows:
                continue

            shard = np.load(foundation_path / str(shard_row["shard"]), allow_pickle=False)
            x = shard["x"].astype(np.float32)
            mask = shard["mask"].astype(bool)
            x_norm = _normalize_chunks(x, mask, mean, std)
            for start in range(0, x_norm.shape[0], batch_size):
                stop = min(start + batch_size, x_norm.shape[0])
                x_tensor = torch.tensor(x_norm[start:stop], dtype=torch.float32, device=torch_device)
                mask_tensor = torch.tensor(mask[start:stop], dtype=torch.bool, device=torch_device)
                hidden = model(x_tensor, mask_tensor).hidden.detach().cpu().numpy()
                _pool_chunk_hidden(
                    hidden=hidden,
                    chunk_offset=start,
                    run_windows=run_windows,
                    grid_hz=grid_hz,
                    chunk_seconds=chunk_seconds,
                    stride_seconds=stride_seconds,
                    embedding_sum=embedding_sum,
                    embedding_steps=embedding_steps,
                )

            processed_shards += 1
            if log_interval > 0 and (processed_shards == 1 or processed_shards % log_interval == 0):
                covered = int(np.sum(embedding_steps > 0))
                print(
                    f"processed_shards={processed_shards} covered_rows={covered}/{len(df)}",
                    flush=True,
                )

    embeddings = np.divide(
        embedding_sum,
        embedding_steps[:, None],
        out=np.zeros_like(embedding_sum),
        where=embedding_steps[:, None] > 0,
    ).astype(np.float32)
    embedding_columns = {
        f"foundation__{event_flag}__dim{dim:03d}": embeddings[:, dim]
        for dim in range(d_model)
    }
    embedding_columns[f"quality__foundation__{event_flag}__n_steps"] = embedding_steps.astype(np.float32)
    embedding_columns[f"quality__foundation__{event_flag}__has_data"] = (embedding_steps > 0).astype(np.float32)
    df = pd.concat([df, pd.DataFrame(embedding_columns, index=df.index)], axis=1)

    _write_supervised_table(df, output_path)
    print(
        f"Saved {len(df)} rows × {len(df.columns)} columns with {d_model} foundation dims → {output_path}",
        flush=True,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Add pretrained foundation embeddings to a supervised MUMT-v2 table.")
    parser.add_argument("--pickle", required=True, help="Input supervised MUMT-v2 .pkl/.csv.")
    parser.add_argument("--foundation_dir", required=True, help="Directory produced by foundation_preprocess_v2.py.")
    parser.add_argument("--checkpoint", required=True, help="foundation_model_best.pt or foundation_model_last.pt.")
    parser.add_argument("--output", required=True, help="Output supervised .pkl/.csv with foundation columns.")
    parser.add_argument("--event_flag", default="video", help="AFFEC event window to pool over; default: video.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--log_interval", type=int, default=20, help="Print progress every N processed shards.")
    args = parser.parse_args()

    add_foundation_embeddings(
        pickle_path=args.pickle,
        foundation_dir=args.foundation_dir,
        checkpoint=args.checkpoint,
        output=args.output,
        event_flag=args.event_flag,
        batch_size=args.batch_size,
        device=args.device,
        max_rows=args.max_rows,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
