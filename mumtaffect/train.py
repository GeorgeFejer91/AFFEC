#!/usr/bin/env python3
"""
mumtaffect/train.py
===================
Subject-disjoint K-fold cross-validation training for AFFECMultiTaskModel.

Key improvements over original MuMTAffect multiphase_simple.py:
  1. Subject-disjoint folds (GroupKFold on participant) — no data leakage
  2. Configurable number of CV folds (default: 5)
  3. Per-fold StandardScaler fitted only on training split
  4. 4-phase training schedule preserved from original
  5. Comprehensive CSV results (per-fold + aggregate)
  6. Grid search over modality ablations and branch configs

Usage:
    python mumtaffect/train.py --pickle data/raw/dataset_enhanced.pkl --folds 5

Quick smoke-test (2 participants, 2 folds):
    python mumtaffect/train.py --pickle data/raw/dataset_enhanced.pkl --folds 2 --max_participants 2
"""

import argparse, io, os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, r2_score, accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mumtaffect.model   import (AFFECMultiTaskModel, EpsilonInsensitiveLoss,
                                 compute_emotion_loss)
from mumtaffect.dataset import (MultiModalDataset, GAZE_COLS, PUPIL_COLS,
                                  AU_COLS, GSR_COLS, CURSOR_COLS,
                                  process_modal_data, COMMON_FLAG_CATEGORIES)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SELECTED_EMOTIONS = ["felt_arousal", "felt_valance", "preceived_arousal", "preceived_valance"]
EMOTION_NAMES     = ["felt_a", "felt_v", "perceived_a", "perceived_v"]
PERSONALITY_COLS  = ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Grid search configs (mirrors original MuMTAffect)
# ─────────────────────────────────────────────────────────────────────────────

MODALITY_GRID = [
    {"eye": True,  "pupil": True,  "au": True,  "gsr": True,  "cursor": True,  "eeg": True,
     "label": "full"},
    {"eye": False, "pupil": False, "au": True,  "gsr": True,  "cursor": False, "eeg": False,
     "label": "au_gsr"},
    {"eye": True,  "pupil": True,  "au": False, "gsr": True,  "cursor": False, "eeg": False,
     "label": "eye_gsr"},
    {"eye": True,  "pupil": True,  "au": True,  "gsr": True,  "cursor": False, "eeg": False,
     "label": "no_eeg_cursor"},
]

BRANCH_GRID = [
    {"branch": "emotion_personality",        "use_gender": False},
    {"branch": "emotion_personality_gender", "use_gender": True},
    {"branch": "emotion_only",               "use_gender": False},
]

STIM_EMO_GRID = [True, False]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: infer model input dimensions from dataset sample
# ─────────────────────────────────────────────────────────────────────────────

def _infer_dims(df: pd.DataFrame) -> dict:
    row = df.iloc[0]
    def _seq_dim(modal_df, cols, flags):
        arr = process_modal_data(modal_df, cols, flags)
        return arr.shape[1] if arr.ndim == 2 else 1

    eye_dim   = _seq_dim(row["Eye_Data"],  GAZE_COLS,   COMMON_FLAG_CATEGORIES)
    pupil_dim = _seq_dim(row["Eye_Data"],  PUPIL_COLS,  COMMON_FLAG_CATEGORIES)
    au_dim    = _seq_dim(row["AUs"],       AU_COLS,     COMMON_FLAG_CATEGORIES)
    gsr_dim   = _seq_dim(row["Shimmer"],   GSR_COLS,    COMMON_FLAG_CATEGORIES)
    # Cursor dim: compute from an actual sample row if available, else fall back to CURSOR_COLS formula
    if "Cursor" in df.columns and isinstance(row.get("Cursor"), pd.DataFrame) and not row["Cursor"].empty:
        cursor_dim = _seq_dim(row["Cursor"], CURSOR_COLS, COMMON_FLAG_CATEGORIES)
    else:
        # onset(1) + CX(1) + CY(1) + CS(1) + len(COMMON_FLAG_CATEGORIES) flag dummies
        cursor_dim = 4 + len(COMMON_FLAG_CATEGORIES)

    from mumtaffect.dataset import flatten_dict, flatten_shimmer_features, eeg_feature_vector
    eye_feat   = len(flatten_dict(row.get("Eye_Data_features", {})))
    au_feat    = len(flatten_dict(row.get("AUs_features", {})))
    shim_feat  = len(flatten_shimmer_features(row))
    eeg_cols   = sorted([c for c in df.columns if c.startswith("eeg_")])
    eeg_dim    = len(eeg_cols)
    cursor_feat = len(flatten_dict(row.get("Cursor_features", {}))) if "Cursor_features" in df.columns else 0

    return dict(eye_dim=eye_dim, pupil_dim=pupil_dim, au_dim=au_dim, gsr_dim=gsr_dim,
                cursor_dim=cursor_dim, eye_feat_dim=max(eye_feat, 1),
                au_feat_dim=max(au_feat, 1), shim_feat_dim=max(shim_feat, 1),
                cursor_feat_dim=max(cursor_feat, 1), eeg_dim=eeg_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Per-modality StandardScaler fitting on train split
# ─────────────────────────────────────────────────────────────────────────────

def fit_scalers(df: pd.DataFrame, train_idx: np.ndarray) -> dict:
    scalers = {}
    train_df = df.iloc[train_idx]

    for name, col, desired_cols in [
        ("Eye_Data",  "Eye_Data",  GAZE_COLS),
        ("AUs",       "AUs",       AU_COLS),
        ("Shimmer",   "Shimmer",   GSR_COLS),
    ]:
        seqs = [process_modal_data(row, desired_cols, COMMON_FLAG_CATEGORIES)
                for row in train_df[col]]
        if seqs:
            all_frames = np.vstack(seqs)
            sc = StandardScaler()
            sc.fit(all_frames)
            scalers[name] = sc

    if "Cursor" in train_df.columns:
        seqs = [process_modal_data(row, CURSOR_COLS, COMMON_FLAG_CATEGORIES)
                for row in train_df["Cursor"]
                if isinstance(row, pd.DataFrame) and not row.empty]
        if seqs:
            sc = StandardScaler()
            sc.fit(np.vstack(seqs))
            scalers["Cursor"] = sc

    # Personality z-score normalisation fitted on training participants only.
    # Raw BFI scores span [9, 49] but the model outputs unbounded linear values,
    # so we normalise to zero-mean unit-variance here.
    pers_vals = train_df[PERSONALITY_COLS].values.astype(np.float32)
    sc_pers = StandardScaler()
    sc_pers.fit(pers_vals)
    scalers["personality"] = sc_pers

    return scalers


# ─────────────────────────────────────────────────────────────────────────────
# Training phases
# ─────────────────────────────────────────────────────────────────────────────

def _personality_target(batch, device):
    """Return continuous personality targets (B, 5) from batch personality field."""
    # batch[6] = personality (B, 5) float
    return batch[6].to(device)


def _emotion_target(batch, device):
    """Return emotion class targets (B, n_tasks) long from batch."""
    return batch[7].to(device)   # emotion_binned


def _gender_target(batch, device):
    return batch[12].to(device)


def _forward(model, batch, use_stim_emo):
    (eye_seq, pupil_seq, au_seq, gsr_seq, cursor_seq, stim_emo,
     personality, emotion_binned, user_id,
     eye_feats, au_feats, shim_feats, gender,
     eeg_feats, cursor_feats) = batch

    return model(
        eye_seq, pupil_seq, au_seq, gsr_seq, cursor_seq,
        stim_emo if use_stim_emo else torch.zeros_like(stim_emo),
        personality,
        eye_feats, au_feats, shim_feats, eeg_feats, cursor_feats,
    )


def _eval_r2(model, loader, device) -> float:
    """Evaluate personality R² over a DataLoader (no_grad)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            pp, _, _, _ = _forward(model, batch, True)
            preds.append(pp.cpu().numpy())
            trues.append(_personality_target(batch, device).cpu().numpy())
    return r2_score(np.vstack(trues), np.vstack(preds))


def phase1_personality(model, loader, epochs, device, patience=5, r2_thresh=0.30,
                       r2_eval_freq=3):
    """Phase 1: freeze emotion head, warm up personality branch.

    r2_eval_freq: how often to run the R² evaluation (every N epochs).
    Running R² requires a full extra forward pass over the training set —
    reducing its frequency gives ~2× speedup for this phase at negligible
    early-stopping accuracy cost.
    """
    # Freeze emotion head
    for name, p in model.named_parameters():
        if "emotion" in name:
            p.requires_grad_(False)
    for p in model.parameters():
        if not any(n in str(p) for n in ["emotion"]):
            p.requires_grad_(True)

    opt     = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    loss_fn = EpsilonInsensitiveLoss(epsilon=0.0)
    best_r2, no_improve, r2 = -np.inf, 0, -np.inf

    for ep in range(epochs):
        model.train()
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            opt.zero_grad()
            persona_pred, _, _, _ = _forward(model, batch, True)
            loss = loss_fn(persona_pred, _personality_target(batch, device))
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # R² evaluation every r2_eval_freq epochs
        if (ep + 1) % r2_eval_freq == 0 or ep == epochs - 1:
            r2 = _eval_r2(model, loader, device)
            if r2 > best_r2:
                best_r2 = r2
                no_improve = 0
            else:
                no_improve += 1
            if r2 >= r2_thresh or no_improve >= patience:
                break

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad_(True)


def phase2_joint(model, loader, val_loader, epochs, device, alpha=0.4,
                 use_stim_emo=True, use_gender=False):
    """Phase 2: joint multitask training."""
    opt = optim.Adam([
        {"params": [p for n, p in model.named_parameters() if "emotion" not in n and "persona" not in n], "lr": 8e-4},
        {"params": [p for n, p in model.named_parameters() if "persona" in n],  "lr": 5e-5},
        {"params": [p for n, p in model.named_parameters() if "emotion" in n],  "lr": 5e-4},
    ])
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    loss_p = EpsilonInsensitiveLoss(epsilon=0.0)

    best_val_f1 = -np.inf
    best_state  = None

    pbar = tqdm(range(epochs), desc="  Ph2 joint", leave=False, unit="ep")
    for ep in pbar:
        model.train()
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            opt.zero_grad()
            persona_pred, _, emo_logits, gender_logits = _forward(model, batch, use_stim_emo)
            tgt_p = _personality_target(batch, device)
            tgt_e = _emotion_target(batch, device)

            lp = loss_p(persona_pred, tgt_p)
            le = compute_emotion_loss(emo_logits, tgt_e, gamma=2.0)
            loss = alpha * lp + (1 - alpha) * le

            if use_gender and gender_logits is not None:
                tgt_g = _gender_target(batch, device)
                lg = nn.CrossEntropyLoss()(gender_logits, tgt_g)
                loss = loss + 0.1 * lg

            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validation F1
        val_f1 = _eval_emotion_f1(model, val_loader, device, use_stim_emo)
        pbar.set_postfix(val_f1=f"{val_f1:.3f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_val_f1


def phase3_personality_refine(model, loader, epochs, device, patience=5, r2_thresh=0.95):
    """Phase 3: refine personality branch only (R² eval every r2_eval_freq epochs)."""
    for name, p in model.named_parameters():
        p.requires_grad_("persona" in name)

    opt     = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = EpsilonInsensitiveLoss(epsilon=0.0)
    best_r2, no_improve, r2 = -np.inf, 0, -np.inf
    r2_eval_freq = 3

    for ep in range(epochs):
        model.train()
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            opt.zero_grad()
            pp, _, _, _ = _forward(model, batch, True)
            loss = loss_fn(pp, _personality_target(batch, device))
            if not torch.isnan(loss):
                loss.backward()
                opt.step()
        sched.step()

        if (ep + 1) % r2_eval_freq == 0 or ep == epochs - 1:
            r2 = _eval_r2(model, loader, device)
            no_improve = 0 if r2 > best_r2 else no_improve + 1
            best_r2 = max(best_r2, r2)
            if r2 >= r2_thresh or no_improve >= patience:
                break

    for p in model.parameters():
        p.requires_grad_(True)


def phase4_emotion_finetune(model, loader, val_loader, epochs, device,
                             use_stim_emo=True):
    """Phase 4: fine-tune emotion heads only."""
    for name, p in model.named_parameters():
        p.requires_grad_("emotion" in name or "trial_fc" in name)

    opt   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_val_f1 = -np.inf
    best_state  = None

    pbar = tqdm(range(epochs), desc="  Ph4 emo  ", leave=False, unit="ep")
    for ep in pbar:
        model.train()
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            opt.zero_grad()
            _, _, emo_logits, _ = _forward(model, batch, use_stim_emo)
            loss = compute_emotion_loss(emo_logits, _emotion_target(batch, device), gamma=2.0)
            if not torch.isnan(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        val_f1 = _eval_emotion_f1(model, val_loader, device, use_stim_emo)
        pbar.set_postfix(val_f1=f"{val_f1:.3f}")
        sched.step(-val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    for p in model.parameters():
        p.requires_grad_(True)
    return best_val_f1


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_emotion_f1(model, loader, device, use_stim_emo) -> float:
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            _, _, emo_logits, _ = _forward(model, batch, use_stim_emo)
            preds = emo_logits.argmax(dim=-1).cpu().numpy()  # (B, n_tasks)
            trues = _emotion_target(batch, device).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(trues)
    P = np.vstack(all_preds)
    T = np.vstack(all_trues)
    f1s = [f1_score(T[:, t], P[:, t], average="macro", zero_division=0)
           for t in range(T.shape[1])]
    return float(np.mean(f1s))


def evaluate(model, loader, device, use_stim_emo) -> dict:
    model.eval()
    e_preds, e_trues = [], []
    p_preds, p_trues = [], []
    g_preds, g_trues = [], []
    uid_list = []

    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            pp, _, emo_logits, gender_logits = _forward(model, batch, use_stim_emo)
            e_preds.append(emo_logits.argmax(dim=-1).cpu().numpy())
            e_trues.append(_emotion_target(batch, device).cpu().numpy())
            p_preds.append(pp.cpu().numpy())
            p_trues.append(_personality_target(batch, device).cpu().numpy())
            uid_list.append(batch[8].cpu().numpy())   # user_id integers
            if gender_logits is not None:
                g_preds.append(gender_logits.argmax(dim=-1).cpu().numpy())
                g_trues.append(_gender_target(batch, device).cpu().numpy())

    EP = np.vstack(e_preds);  ET = np.vstack(e_trues)
    PP = np.vstack(p_preds);  PT = np.vstack(p_trues)
    UID = np.concatenate(uid_list)

    results = {}
    for t, name in enumerate(EMOTION_NAMES):
        results[f"f1_macro_{name}"]   = float(f1_score(ET[:,t], EP[:,t], average="macro",    zero_division=0))
        results[f"f1_weighted_{name}"] = float(f1_score(ET[:,t], EP[:,t], average="weighted", zero_division=0))
        results[f"accuracy_{name}"]   = float(accuracy_score(ET[:,t], EP[:,t]))
    results["emotion_macro_f1_mean"] = float(np.mean([results[f"f1_macro_{n}"] for n in EMOTION_NAMES]))

    # Trial-level R² (for reference)
    try:
        r2_vals = r2_score(PT, PP, multioutput="raw_values")
        for i, pc in enumerate(PERSONALITY_COLS):
            results[f"r2_{pc}"] = float(r2_vals[i])
        results["personality_r2_mean"] = float(np.mean(r2_vals))
    except Exception:
        results["personality_r2_mean"] = np.nan

    # Participant-level R²: average predictions across all trials of each test
    # participant, then compare to that participant's true personality.
    # This is the correct unit of analysis — personality is constant per person,
    # so per-trial noise should be averaged out before scoring.
    try:
        uids = np.unique(UID)
        part_pred = np.array([PP[UID == u].mean(0) for u in uids])
        part_true = np.array([PT[UID == u].mean(0) for u in uids])
        if len(uids) >= 2:
            r2_part = r2_score(part_true, part_pred, multioutput="raw_values")
            for i, pc in enumerate(PERSONALITY_COLS):
                results[f"part_r2_{pc}"] = float(r2_part[i])
            results["part_personality_r2_mean"] = float(np.mean(r2_part))
        else:
            results["part_personality_r2_mean"] = np.nan
    except Exception:
        results["part_personality_r2_mean"] = np.nan

    if g_preds:
        GP = np.concatenate(g_preds); GT = np.concatenate(g_trues)
        results["gender_f1"] = float(f1_score(GT, GP, average="macro", zero_division=0))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Subject-disjoint K-fold cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold_cv(df: pd.DataFrame, cfg: dict, n_folds: int,
                 dims: dict, out_dir: str, args) -> list:
    """
    Subject-disjoint K-fold CV.

    Groups = participant IDs; GroupKFold ensures no participant appears
    in both train and test within any fold.
    """
    groups = df["user"].values
    gkf    = GroupKFold(n_splits=n_folds)
    user2idx = {u: i for i, u in enumerate(df["user"].unique())}

    fold_results = []
    label = f"{cfg['modality']['label']}_{cfg['branch']}{'_stim' if cfg['use_stim_emo'] else ''}"

    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n  ── Fold {fold+1}/{n_folds}  ({label}) ──", flush=True)

        # Split train_val → 85% train / 15% val (still subject-disjoint by above)
        train_users  = list(set(groups[train_val_idx]))
        n_val_users  = max(1, int(0.15 * len(train_users)))
        rng          = np.random.default_rng(42 + fold)
        val_users    = set(rng.choice(train_users, n_val_users, replace=False))
        train_idx    = np.array([i for i in train_val_idx if groups[i] not in val_users])
        val_idx      = np.array([i for i in train_val_idx if groups[i] in val_users])

        scalers = fit_scalers(df, train_idx)

        def mk_ds(idx, augment=False):
            return MultiModalDataset(
                df.iloc[idx].reset_index(drop=True),
                SELECTED_EMOTIONS, user2idx, scalers, DEVICE, augment=augment,
            )
        train_ds = mk_ds(train_idx, augment=True)
        val_ds   = mk_ds(val_idx,   augment=False)
        test_ds  = mk_ds(test_idx,  augment=False)

        bs = args.batch_size
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

        # Build model
        model = AFFECMultiTaskModel(
            **{k: v for k, v in dims.items()},
            n_tasks    = len(SELECTED_EMOTIONS),
            n_classes  = 3,
            use_gender   = cfg["use_gender"],
            use_stim_emo = cfg["use_stim_emo"],
            use_eeg      = dims["eeg_dim"] > 0 and cfg["modality"]["eeg"],
            use_cursor   = cfg["modality"]["cursor"],
            dropout      = 0.1,
        ).to(DEVICE)

        # 4-phase training
        t0 = time.time()
        phase1_personality(model, train_loader, epochs=10,  device=DEVICE)
        phase2_joint(model, train_loader, val_loader, epochs=args.phase2_epochs,
                     device=DEVICE, use_stim_emo=cfg["use_stim_emo"],
                     use_gender=cfg["use_gender"])
        phase3_personality_refine(model, train_loader, epochs=15, device=DEVICE)
        phase4_emotion_finetune(model, train_loader, val_loader, epochs=args.phase4_epochs,
                                device=DEVICE, use_stim_emo=cfg["use_stim_emo"])
        elapsed = time.time() - t0

        # Evaluate
        test_metrics = evaluate(model, test_loader, DEVICE, cfg["use_stim_emo"])
        val_metrics  = evaluate(model, val_loader,  DEVICE, cfg["use_stim_emo"])

        row = {
            "config": label,
            "fold":   fold,
            "modality": cfg["modality"]["label"],
            "branch":   cfg["branch"],
            "use_stim_emo": cfg["use_stim_emo"],
            "n_train": len(train_idx),
            "n_val":   len(val_idx),
            "n_test":  len(test_idx),
            "train_time_s": round(elapsed, 1),
            **{f"val_{k}":  v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        fold_results.append(row)

        print(f"    test emotion_macro_f1={test_metrics['emotion_macro_f1_mean']:.3f}"
              f"  part_personality_r2={test_metrics.get('part_personality_r2_mean', float('nan')):.3f}"
              f"  trial_r2={test_metrics.get('personality_r2_mean', float('nan')):.3f}"
              f"  ({elapsed:.0f}s)", flush=True)

        # Save model checkpoint
        ckpt_path = os.path.join(out_dir, f"model_{label}_fold{fold}.pt")
        torch.save(model.state_dict(), ckpt_path)

    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Force line-buffered stdout so output appears in log files immediately
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

    parser = argparse.ArgumentParser(description="MuMTAffect — subject-disjoint K-fold CV training")
    parser.add_argument("--pickle",           type=str,  required=True,
                        help="Path to dataset_enhanced.pkl")
    parser.add_argument("--folds",            type=int,  default=5,
                        help="Number of CV folds (default 5)")
    parser.add_argument("--batch_size",       type=int,  default=32)
    parser.add_argument("--phase2_epochs",    type=int,  default=60,
                        help="Max epochs for joint phase 2")
    parser.add_argument("--phase4_epochs",    type=int,  default=30,
                        help="Max epochs for emotion fine-tune phase 4")
    parser.add_argument("--max_participants", type=int,  default=None,
                        help="Limit to N participants (for quick testing)")
    parser.add_argument("--out_dir",          type=str,  default="mumtaffect/grid_results")
    parser.add_argument("--grid_modality",    type=int,  default=0,
                        help="Which modality config to run (0=full, 1=au_gsr, …)")
    parser.add_argument("--all_configs",      action="store_true",
                        help="Run all grid configurations (slow)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t_start = time.time()
    print(f"Loading pickle: {args.pickle}", flush=True)
    df = pd.read_pickle(args.pickle)
    print(f"Loaded {len(df)} trials from {df['user'].nunique()} participants  "
          f"({time.time()-t_start:.1f}s)", flush=True)

    if args.max_participants:
        users = df["user"].unique()[:args.max_participants]
        df    = df[df["user"].isin(users)].reset_index(drop=True)
        print(f"Restricted to {len(df)} trials ({args.max_participants} participants)", flush=True)

    dims = _infer_dims(df)
    print("Inferred model input dimensions:", {k: v for k, v in dims.items() if v > 0}, flush=True)

    # Build grid
    if args.all_configs:
        grid = [
            {"modality": mod, "branch": br["branch"], "use_gender": br["use_gender"],
             "use_stim_emo": se}
            for mod in MODALITY_GRID
            for br  in BRANCH_GRID
            for se  in STIM_EMO_GRID
        ]
    else:
        mod = MODALITY_GRID[args.grid_modality]
        grid = [
            {"modality": mod, "branch": br["branch"], "use_gender": br["use_gender"],
             "use_stim_emo": se}
            for br in BRANCH_GRID
            for se in STIM_EMO_GRID
        ]

    print(f"\nRunning {len(grid)} configurations × {args.folds} folds on {DEVICE}", flush=True)

    all_results = []
    for cfg_idx, cfg in enumerate(grid):
        label = f"{cfg['modality']['label']}_{cfg['branch']}{'_stim' if cfg['use_stim_emo'] else ''}"
        print(f"\n{'═'*60}\nConfig {cfg_idx+1}/{len(grid)}: {label}\n{'═'*60}")
        fold_results = run_kfold_cv(df, cfg, args.folds, dims, args.out_dir, args)
        all_results.extend(fold_results)

    # Save all per-fold results
    folds_path = os.path.join(args.out_dir, "cv_results.csv")
    pd.DataFrame(all_results).to_csv(folds_path, index=False)
    print(f"\n✓ Per-fold results → {folds_path}")

    # Aggregate mean ± std per config
    fold_df = pd.DataFrame(all_results)
    metric_cols = [c for c in fold_df.columns
                   if c.startswith("test_") and fold_df[c].dtype in (float, np.float64)]
    agg = (
        fold_df.groupby("config")[metric_cols]
        .agg(["mean", "std"])
    )
    agg_path = os.path.join(args.out_dir, "results_aggregate.csv")
    agg.to_csv(agg_path)
    print(f"✓ Aggregate results  → {agg_path}")

    # Pretty-print top configs by emotion F1
    if "test_emotion_macro_f1_mean" in fold_df.columns:
        top = (
            fold_df.groupby("config")["test_emotion_macro_f1_mean"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
            .head(5)
        )
        print("\n── Top 5 configs by emotion macro-F1 ──")
        print(top.to_string(float_format="%.3f"))


if __name__ == "__main__":
    main()
