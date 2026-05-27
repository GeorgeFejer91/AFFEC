#!/usr/bin/env python3
"""
Minimal subject-disjoint training scaffold for MUMT-v2.

This is intentionally smaller than `train.py`: it trains the new event-token
model on `dataset_mumt_v2.pkl`, reports macro-F1/accuracy per task, and exposes
the first few-shot path through support-set user prototypes.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from mumtaffect.dataset_v2 import (
    BIN_TARGETS,
    GAP_TARGETS,
    RAW_TARGETS,
    MUMTV2Dataset,
    collate_v2,
    fit_v2_scaler,
    select_feature_columns,
    split_support_query,
)
from mumtaffect.model_v2 import MUMTV2Loss, MUMTV2Model, TASK_NAMES, compute_class_weights, model_summary
from mumtaffect.profile_features import add_global_label_style_defaults, apply_label_style, fit_label_style


DEFAULT_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = DEFAULT_DEVICE


def log(message: str) -> None:
    print(message, flush=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return DEFAULT_DEVICE
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but not available.")
    return device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_label_style(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_label_style: bool,
    few_shot_k: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Add leakage-safe label-style features.

    Train users get style from train labels. Held-out users get no style in
    zero-shot mode; in few-shot mode query rows get style fitted only on support.
    """
    train_df = train_df.reset_index(drop=True).copy()
    val_df = val_df.reset_index(drop=True).copy()
    test_df = test_df.reset_index(drop=True).copy()

    support_df = None
    query_df = test_df
    if few_shot_k > 0:
        support_idx, query_idx = split_support_query(test_df, k=few_shot_k, seed=seed)
        support_df = test_df.loc[support_idx].reset_index(drop=True)
        query_df = test_df.loc[query_idx].reset_index(drop=True)

    if not use_label_style:
        return train_df, val_df, support_df, query_df

    train_style = fit_label_style(train_df)
    train_df = apply_label_style(train_df, train_style)
    val_df = add_global_label_style_defaults(val_df)

    if support_df is None:
        query_df = add_global_label_style_defaults(query_df)
    else:
        support_style = fit_label_style(support_df)
        support_df = apply_label_style(support_df, support_style)
        query_df = apply_label_style(query_df, support_style)

    return train_df, val_df, support_df, query_df


def make_dataset(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    scaler,
    user2idx: dict[str, int],
) -> MUMTV2Dataset:
    return MUMTV2Dataset(
        dataframe,
        feature_columns=feature_columns,
        scaler=scaler,
        user2idx=user2idx,
        device=DEVICE,
    )


def train_one_epoch(
    model: MUMTV2Model,
    loader: DataLoader,
    loss_fn: MUMTV2Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    n_epochs: int,
    batch_log_interval: int = 0,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0
    n_total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["features"])
        loss, components = loss_fn(
            output,
            labels_bin=batch["labels_bin"],
            labels_raw=batch["labels_raw"],
            labels_gap=batch["labels_gap"],
        )
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
        if batch_log_interval > 0 and (batch_index % batch_log_interval == 0 or batch_index == n_total_batches):
            log(
                f"    train batch {batch_index:03d}/{n_total_batches:03d} "
                f"epoch={epoch}/{n_epochs} "
                f"loss={components.get('total', np.nan):.4f}"
            )

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def build_user_prototypes(model: MUMTV2Model, support_loader: Optional[DataLoader]) -> dict[str, torch.Tensor]:
    if support_loader is None:
        return {}

    model.eval()
    embeddings_by_user: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in support_loader:
            embeddings = model.embed_features(batch["features"])
            for row_index, user in enumerate(batch["user"]):
                embeddings_by_user.setdefault(user, []).append(embeddings[row_index].detach())

    return {
        user: torch.stack(user_embeddings, dim=0).mean(dim=0)
        for user, user_embeddings in embeddings_by_user.items()
        if user_embeddings
    }


def prototype_batch(
    users: list[str],
    prototypes: dict[str, torch.Tensor],
    d_model: int,
) -> Optional[torch.Tensor]:
    if not prototypes:
        return None
    vectors = [
        prototypes.get(user, torch.zeros(d_model, device=DEVICE))
        for user in users
    ]
    return torch.stack(vectors, dim=0)


def evaluate_loss(
    model: MUMTV2Model,
    loader: DataLoader,
    loss_fn: MUMTV2Loss,
    prototypes: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            user_prototype = prototype_batch(batch["user"], prototypes or {}, model.d_model)
            output = model(batch["features"], user_prototype=user_prototype)
            _, components = loss_fn(
                output,
                labels_bin=batch["labels_bin"],
                labels_raw=batch["labels_raw"],
                labels_gap=batch["labels_gap"],
            )
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def evaluate(
    model: MUMTV2Model,
    loader: DataLoader,
    prototypes: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, float]:
    model.eval()
    predicted_bins = []
    true_bins = []
    predicted_raw = []
    true_raw = []
    predicted_gaps = []
    true_gaps = []

    with torch.no_grad():
        for batch in loader:
            user_prototype = prototype_batch(batch["user"], prototypes or {}, model.d_model)
            output = model(batch["features"], user_prototype=user_prototype)
            predicted_bins.append(output.emotion_logits.argmax(dim=-1).cpu().numpy())
            true_bins.append(batch["labels_bin"].cpu().numpy())
            predicted_raw.append(output.rating_raw.cpu().numpy())
            true_raw.append(batch["labels_raw"].cpu().numpy())
            predicted_gaps.append(output.gap_raw.cpu().numpy())
            true_gaps.append(batch["labels_gap"].cpu().numpy())

    bin_predictions = np.vstack(predicted_bins)
    bin_targets = np.vstack(true_bins)
    raw_predictions = np.vstack(predicted_raw)
    raw_targets = np.vstack(true_raw)
    gap_predictions = np.vstack(predicted_gaps)
    gap_targets = np.vstack(true_gaps)

    metrics: dict[str, float] = {}
    task_f1_values = []
    for task_index, task_name in enumerate(TASK_NAMES):
        valid_mask = (bin_targets[:, task_index] >= 0) & (bin_targets[:, task_index] < 3)
        if not np.any(valid_mask):
            continue
        metrics[f"{task_name}_macro_f1"] = float(
            f1_score(
                bin_targets[valid_mask, task_index],
                bin_predictions[valid_mask, task_index],
                average="macro",
                zero_division=0,
            )
        )
        metrics[f"{task_name}_accuracy"] = float(
            accuracy_score(
                bin_targets[valid_mask, task_index],
                bin_predictions[valid_mask, task_index],
            )
        )
        task_f1_values.append(metrics[f"{task_name}_macro_f1"])

    metrics["emotion_macro_f1_mean"] = float(np.mean(task_f1_values)) if task_f1_values else np.nan

    raw_mask = np.isfinite(raw_targets)
    if np.any(raw_mask):
        metrics["raw_rating_mae"] = float(mean_absolute_error(raw_targets[raw_mask], raw_predictions[raw_mask]))

    gap_mask = np.isfinite(gap_targets)
    if np.any(gap_mask):
        metrics["gap_mae"] = float(mean_absolute_error(gap_targets[gap_mask], gap_predictions[gap_mask]))

    return metrics


def run_fold(
    dataframe: pd.DataFrame,
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    args,
    user2idx: dict[str, int],
) -> dict[str, float]:
    groups = dataframe["user"].astype(str).to_numpy()
    train_val_users = np.array(sorted(set(groups[train_val_idx])))
    rng = np.random.default_rng(args.seed + fold)
    n_val_users = max(1, int(len(train_val_users) * args.val_user_fraction))
    val_users = set(rng.choice(train_val_users, size=n_val_users, replace=False))

    train_idx = np.array([row_index for row_index in train_val_idx if groups[row_index] not in val_users])
    val_idx = np.array([row_index for row_index in train_val_idx if groups[row_index] in val_users])

    train_df, val_df, support_df, query_df = prepare_label_style(
        train_df=dataframe.iloc[train_idx],
        val_df=dataframe.iloc[val_idx],
        test_df=dataframe.iloc[test_idx],
        use_label_style=args.use_labelstyle,
        few_shot_k=args.few_shot_k,
        seed=args.seed + fold,
    )

    feature_columns = select_feature_columns(train_df)
    scaler = fit_v2_scaler(train_df, feature_columns)
    train_loader = DataLoader(
        make_dataset(train_df, feature_columns, scaler, user2idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_v2,
    )
    val_loader = DataLoader(
        make_dataset(val_df, feature_columns, scaler, user2idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_v2,
    )
    query_loader = DataLoader(
        make_dataset(query_df, feature_columns, scaler, user2idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_v2,
    )
    support_loader = None
    if support_df is not None and not support_df.empty:
        support_loader = DataLoader(
            make_dataset(support_df, feature_columns, scaler, user2idx),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_v2,
        )

    model = MUMTV2Model(
        feature_columns=feature_columns,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        modality_dropout_p=args.modality_dropout,
        foundation_gate=args.foundation_gate,
        foundation_gate_init=args.foundation_gate_init,
    ).to(DEVICE)

    class_weights = compute_class_weights(train_df.loc[:, BIN_TARGETS].to_numpy()).to(DEVICE)
    loss_fn = MUMTV2Loss(
        class_weights=class_weights,
        emotion_weight=args.emotion_weight,
        rating_weight=args.rating_weight,
        gap_weight=args.gap_weight,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    log(f"Fold {fold}: {len(feature_columns)} features, {len(model.group_names)} tokens")
    if fold == 0:
        token_preview = list(model_summary(feature_columns).items())[:12]
        log(f"Token preview: {token_preview}")

    best_val_f1 = -np.inf
    best_monitor_score = -np.inf
    best_monitor_text = ""
    best_epoch = 0
    best_state = None
    no_improve_epochs = 0
    early_stopped = False
    start_time = time.time()
    for epoch in range(args.epochs):
        log(f"  starting epoch {epoch + 1:03d}/{args.epochs:03d}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            epoch=epoch + 1,
            n_epochs=args.epochs,
            batch_log_interval=args.batch_log_interval,
        )
        val_loss = evaluate_loss(model, val_loader, loss_fn)
        val_metrics = evaluate(model, val_loader)
        val_f1 = val_metrics.get("emotion_macro_f1_mean", np.nan)
        scheduler.step(val_f1 if np.isfinite(val_f1) else -1.0)
        if np.isfinite(val_f1):
            best_val_f1 = max(best_val_f1, val_f1)

        val_loss_total = val_loss.get("total", np.nan)
        if args.early_stop_monitor == "val_loss":
            monitor_score = -val_loss_total if np.isfinite(val_loss_total) else -np.inf
            monitor_text = f"val_loss={val_loss_total:.4f}"
        else:
            monitor_score = val_f1 if np.isfinite(val_f1) else -np.inf
            monitor_text = f"val_f1={val_f1:.3f}"

        if monitor_score > best_monitor_score + args.early_stop_min_delta:
            best_monitor_score = monitor_score
            best_monitor_text = monitor_text
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        metric_due = (
            epoch == 0
            or (epoch + 1) % args.metric_interval == 0
            or (epoch + 1) == args.epochs
        )
        current_lr = optimizer.param_groups[0]["lr"]
        log(
            f"  epoch {epoch + 1:03d}: "
            f"train_loss={train_loss.get('total', np.nan):.4f} "
            f"val_loss={val_loss.get('total', np.nan):.4f} "
            f"train_emotion={train_loss.get('emotion', np.nan):.4f} "
            f"val_emotion={val_loss.get('emotion', np.nan):.4f} "
            f"lr={current_lr:.2e}"
        )
        if metric_due:
            log(
                f"    val_f1={val_f1:.3f} "
                f"felt_a={val_metrics.get('felt_arousal_macro_f1', np.nan):.3f} "
                f"felt_v={val_metrics.get('felt_valence_macro_f1', np.nan):.3f} "
                f"perc_a={val_metrics.get('perceived_arousal_macro_f1', np.nan):.3f} "
                f"perc_v={val_metrics.get('perceived_valence_macro_f1', np.nan):.3f} "
                f"raw_mae={val_metrics.get('raw_rating_mae', np.nan):.3f} "
                f"gap_mae={val_metrics.get('gap_mae', np.nan):.3f}"
            )
        if (
            args.early_stop_patience > 0
            and epoch + 1 >= args.min_epochs
            and no_improve_epochs >= args.early_stop_patience
        ):
            early_stopped = True
            log(
                f"  early stopping at epoch {epoch + 1:03d}: "
                f"best_epoch={best_epoch:03d} monitor={args.early_stop_monitor} "
                f"best={best_monitor_text} no_improve={no_improve_epochs}"
            )
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(DEVICE) for key, value in best_state.items()})

    prototypes = build_user_prototypes(model, support_loader)
    test_metrics = evaluate(model, query_loader, prototypes=prototypes)
    elapsed = time.time() - start_time

    checkpoint = {
        "model_state": model.state_dict(),
        "feature_columns": feature_columns,
        "group_names": model.group_names,
        "args": vars(args),
    }
    checkpoint_path = Path(args.out_dir) / f"mumt_v2_fold{fold}.pt"
    torch.save(checkpoint, checkpoint_path)

    return {
        "fold": fold,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_support": 0 if support_df is None else len(support_df),
        "n_query": len(query_df),
        "n_features": len(feature_columns),
        "n_tokens": len(model.group_names),
        "best_val_emotion_macro_f1_mean": float(best_val_f1),
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "early_stopped": early_stopped,
        "train_time_s": round(elapsed, 1),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MUMT-v2 event-token model.")
    parser.add_argument("--pickle", default="data/raw/dataset_mumt_v2.pkl", help="Path to MUMT-v2 dataframe pickle.")
    parser.add_argument("--out_dir", default="mumtaffect/grid_results_v2", help="Output directory.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_participants", type=int, default=None)
    parser.add_argument("--few_shot_k", type=int, default=0, help="Support trials per held-out user.")
    parser.add_argument("--use_labelstyle", action="store_true", help="Use leakage-safe train/support label-style features.")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--modality_dropout", type=float, default=0.15)
    parser.add_argument("--foundation_gate", action="store_true", help="Use a learnable scalar gate for foundation tokens.")
    parser.add_argument("--foundation_gate_init", type=float, default=-2.0, help="Initial foundation gate logit; -2 ≈ 0.12.")
    parser.add_argument("--emotion_weight", type=float, default=1.0)
    parser.add_argument("--rating_weight", type=float, default=0.25)
    parser.add_argument("--gap_weight", type=float, default=0.20)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--val_user_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--metric_interval", type=int, default=5, help="Print validation F1/MAE every N epochs.")
    parser.add_argument("--batch_log_interval", type=int, default=0, help="Print training batch loss every N batches; 0 disables batch logs.")
    parser.add_argument("--early_stop_patience", type=int, default=6, help="Stop after N epochs without monitor improvement; 0 disables.")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3, help="Minimum monitor improvement.")
    parser.add_argument("--early_stop_monitor", choices=("val_f1", "val_loss"), default="val_f1")
    parser.add_argument("--min_epochs", type=int, default=5)
    args = parser.parse_args()

    global DEVICE
    DEVICE = resolve_device(args.device)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    dataframe = pd.read_pickle(args.pickle)
    if args.max_participants is not None:
        users = dataframe["user"].astype(str).drop_duplicates().iloc[:args.max_participants]
        dataframe = dataframe[dataframe["user"].astype(str).isin(users)].reset_index(drop=True)

    log(f"Loaded {len(dataframe)} trials from {dataframe['user'].nunique()} users on {DEVICE}")
    log(f"Targets: raw={RAW_TARGETS}, bins={BIN_TARGETS}, gaps={GAP_TARGETS}")

    groups = dataframe["user"].astype(str).to_numpy()
    user2idx = {user: user_index for user_index, user in enumerate(sorted(set(groups)))}
    splitter = GroupKFold(n_splits=args.folds)

    fold_rows = []
    for fold, (train_val_idx, test_idx) in enumerate(splitter.split(dataframe, groups=groups)):
        fold_rows.append(run_fold(dataframe, train_val_idx, test_idx, fold, args, user2idx))

    results = pd.DataFrame(fold_rows)
    results_path = Path(args.out_dir) / "cv_results_v2.csv"
    results.to_csv(results_path, index=False)

    metric_cols = [column for column in results.columns if column.startswith("test_")]
    aggregate = results[metric_cols].agg(["mean", "std"]).T
    aggregate_path = Path(args.out_dir) / "results_aggregate_v2.csv"
    aggregate.to_csv(aggregate_path)

    log(f"Saved per-fold results: {results_path}")
    log(f"Saved aggregate results: {aggregate_path}")


if __name__ == "__main__":
    main()
