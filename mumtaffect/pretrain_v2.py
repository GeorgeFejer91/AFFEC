"""
Train the MUMT-v2 full-run self-supervised foundation model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mumtaffect.foundation_dataset_v2 import MUMTFoundationDataset, collate_foundation
from mumtaffect.foundation_model_v2 import MUMTFoundationModel, foundation_ssl_loss, make_ssl_inputs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_requested_modalities(value: str) -> Optional[list[str]]:
    if not value or value.lower() == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def split_shards(data_dir: str, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    manifest = pd.read_csv(Path(data_dir) / "manifest.csv")
    rng = np.random.default_rng(seed)
    if len(manifest) == 1:
        return [0], [0]

    users = np.array(sorted(manifest["user"].astype(str).unique()))
    if len(users) > 1:
        n_val_users = max(1, int(round(len(users) * val_fraction)))
        n_val_users = min(n_val_users, len(users) - 1)
        val_users = set(rng.choice(users, size=n_val_users, replace=False).tolist())
        train_indices = manifest.index[~manifest["user"].astype(str).isin(val_users)].tolist()
        val_indices = manifest.index[manifest["user"].astype(str).isin(val_users)].tolist()
        if train_indices and val_indices:
            return train_indices, val_indices

    indices = np.arange(len(manifest))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(indices) * val_fraction)))
    n_val = min(n_val, len(indices) - 1)
    return indices[n_val:].tolist(), indices[:n_val].tolist()


def run_epoch(
    model: MUMTFoundationModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    modality_slices: dict,
    requested_modalities: Optional[Sequence[str]],
    mask_modality_p: float,
    mask_time_p: float,
    next_horizon: int,
    recon_weight: float,
    next_weight: float,
    epoch: int,
    phase: str,
    batch_log_interval: int,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "recon_loss": 0.0, "next_loss": 0.0, "n": 0}

    total_batches = len(loader)
    started_at = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        observed_mask = batch["mask"].to(device)
        masked_x, target_mask = make_ssl_inputs(
            x=x,
            observed_mask=observed_mask,
            modality_slices=modality_slices,
            mask_modality_p=mask_modality_p,
            mask_time_p=mask_time_p,
            requested_modalities=requested_modalities,
        )

        with torch.set_grad_enabled(train):
            outputs = model(masked_x, observed_mask)
            loss, parts = foundation_ssl_loss(
                outputs=outputs,
                target_x=x,
                target_mask=target_mask,
                observed_mask=observed_mask,
                next_horizon=next_horizon,
                recon_weight=recon_weight,
                next_weight=next_weight,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = x.size(0)
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["recon_loss"] += float(parts["recon_loss"].cpu()) * batch_size
        totals["next_loss"] += float(parts["next_loss"].cpu()) * batch_size
        totals["n"] += batch_size

        if batch_log_interval > 0 and (batch_index == 1 or batch_index % batch_log_interval == 0 or batch_index == total_batches):
            denom = max(totals["n"], 1)
            elapsed = time.perf_counter() - started_at
            print(
                f"epoch {epoch:03d} {phase} batch {batch_index:04d}/{total_batches:04d} "
                f"loss={totals['loss'] / denom:.5f} recon={totals['recon_loss'] / denom:.5f} "
                f"next={totals['next_loss'] / denom:.5f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    denom = max(totals["n"], 1)
    return {key: totals[key] / denom for key in ("loss", "recon_loss", "next_loss")}


def train_foundation(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.data_dir) / "schema.json", "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    modality_slices = schema["modality_slices"]
    requested_modalities = parse_requested_modalities(args.target_modalities)
    if requested_modalities:
        missing = sorted(set(requested_modalities) - set(modality_slices))
        if missing:
            raise ValueError(f"Requested modalities not present in schema: {missing}")

    train_shards, val_shards = split_shards(args.data_dir, args.val_fraction, args.seed)
    train_dataset = MUMTFoundationDataset(args.data_dir, shard_indices=train_shards, normalize=True)
    val_dataset = MUMTFoundationDataset(args.data_dir, shard_indices=val_shards, normalize=True)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_foundation,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_foundation,
    )

    model = MUMTFoundationModel(
        n_features=len(schema["feature_names"]),
        max_seq_len=int(schema["chunk_seconds"] * schema["grid_hz"]),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = out_dir / "training_log.csv"
    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_recon_loss",
                "train_next_loss",
                "val_loss",
                "val_recon_loss",
                "val_next_loss",
            ],
        )
        writer.writeheader()
        handle.flush()

        print(
            f"Device={device} train_chunks={len(train_dataset)} val_chunks={len(val_dataset)} "
            f"features={len(schema['feature_names'])} target_modalities={requested_modalities or 'all'}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                modality_slices=modality_slices,
                requested_modalities=requested_modalities,
                mask_modality_p=args.mask_modality_p,
                mask_time_p=args.mask_time_p,
                next_horizon=args.next_horizon,
                recon_weight=args.recon_weight,
                next_weight=args.next_weight,
                epoch=epoch,
                phase="train",
                batch_log_interval=args.batch_log_interval,
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                modality_slices=modality_slices,
                requested_modalities=requested_modalities,
                mask_modality_p=args.mask_modality_p,
                mask_time_p=args.mask_time_p,
                next_horizon=args.next_horizon,
                recon_weight=args.recon_weight,
                next_weight=args.next_weight,
                epoch=epoch,
                phase="val",
                batch_log_interval=args.batch_log_interval,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_recon_loss": train_metrics["recon_loss"],
                "train_next_loss": train_metrics["next_loss"],
                "val_loss": val_metrics["loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_next_loss": val_metrics["next_loss"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_loss={row['train_loss']:.5f} train_recon={row['train_recon_loss']:.5f} train_next={row['train_next_loss']:.5f} "
                f"val_loss={row['val_loss']:.5f} val_recon={row['val_recon_loss']:.5f} val_next={row['val_next_loss']:.5f}",
                flush=True,
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_metrics = row
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "schema": schema,
                        "args": vars(args),
                        "best_metrics": best_metrics,
                    },
                    out_dir / "foundation_model_best.pt",
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "schema": schema,
            "args": vars(args),
            "best_metrics": best_metrics,
        },
        out_dir / "foundation_model_last.pt",
    )
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a MUMT-v2 full-run self-supervised foundation model.")
    parser.add_argument("--data_dir", required=True, help="Directory produced by foundation_preprocess_v2.py.")
    parser.add_argument("--out_dir", required=True, help="Output directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--mask_modality_p", type=float, default=0.35)
    parser.add_argument("--mask_time_p", type=float, default=0.05)
    parser.add_argument("--target_modalities", default="all", help="Comma list such as gsr,eeg, or all.")
    parser.add_argument("--next_horizon", type=int, default=1, help="Grid steps ahead. At 20 Hz, 1 step = 50 ms.")
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--next_weight", type=float, default=0.5)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_log_interval", type=int, default=50, help="Print running train/val losses every N batches.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    args = parser.parse_args()

    train_foundation(args)


if __name__ == "__main__":
    main()
