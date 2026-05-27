"""
mumtaffect/model.py
===================
Neural architectures for multimodal multitask affective computing.

Architecture: AdvancedConcatFusionMultiModalModelTransformerAttention_v2 (enhanced)

Enhancements over original MuMTAffect:
  • EEGFeatureEncoder — processes 315-dim band-power features alongside other trial features
  • CursorEncoder — lightweight MLP for cursor trajectory stats
  • Modality dropout at training time for robustness
  • Separate self-attention emotion head (v2 variant)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class EpsilonInsensitiveLoss(nn.Module):
    """SVR-style loss: squared penalty only when |error| > epsilon."""
    def __init__(self, epsilon: float = 0.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target.float()).abs()
        loss = torch.where(diff < self.epsilon, torch.zeros_like(diff), (diff - self.epsilon) ** 2)
        return loss.mean()


def compute_emotion_loss(
    logits: torch.Tensor,      # (B, n_tasks, n_classes)
    targets: torch.Tensor,     # (B, n_tasks) long
    class_weights: torch.Tensor = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy summed over emotion tasks."""
    B, T, C = logits.shape
    total = torch.tensor(0.0, device=logits.device)
    for t in range(T):
        tgt = targets[:, t]
        mask = (tgt >= 0) & (tgt < C)
        if mask.sum() == 0:
            continue
        log_p = F.log_softmax(logits[mask, t, :], dim=-1)
        log_p = torch.clamp(log_p, min=-100.0)
        p     = log_p.exp().detach()
        w     = (1 - p) ** gamma
        nll   = F.nll_loss(log_p, tgt[mask], weight=class_weights, reduction="none")
        focal = (w.gather(1, tgt[mask].unsqueeze(1)).squeeze(1) * nll).mean()
        if not torch.isnan(focal):
            total = total + focal
    return total / max(T, 1)


def scale_with_sigmoid(x: torch.Tensor, lo: float = 1.0, hi: float = 9.0) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


# ─────────────────────────────────────────────────────────────────────────────
# Attention / pooling modules
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        w = torch.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (x * w).sum(dim=1)                 # (B, D)


class TaskAttention(nn.Module):
    """Learnable query vectors for personality vs emotion extraction."""
    def __init__(self, d_model: int, n_heads: int = 4, n_queries: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, n_queries, D)
        q = self.queries.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Temporal downsampling
# ─────────────────────────────────────────────────────────────────────────────

def downsample_sequence(x: torch.Tensor, target: int) -> torch.Tensor:
    """Downsample time dimension to `target` steps. x: (B, T, D)
    Uses F.interpolate (mode='area') which is MPS-safe unlike adaptive_avg_pool1d."""
    if x.size(1) == target:
        return x
    # (B, T, D) → (B, D, T) → interpolate → (B, D, target) → (B, target, D)
    out = F.interpolate(x.permute(0, 2, 1).float(), size=target, mode="linear", align_corners=False)
    return out.permute(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Modality encoders
# ─────────────────────────────────────────────────────────────────────────────

class ModalityTransformerEncoder(nn.Module):
    """
    Per-modality transformer encoder with projection and positional encoding.

    Key performance optimisation: downsample the raw sequence to ``target_seq_len``
    BEFORE the Transformer layers.  Attention complexity is O(T²), so encoding at
    T=64 instead of T=400 gives a ~15× per-encoder speedup while keeping the
    fusion transformer input unchanged.
    """
    def __init__(self, in_dim: int, d_model: int, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, max_len: int = 512, target_seq_len: int = 64):
        super().__init__()
        self.target_seq_len = target_seq_len
        self.proj = nn.Linear(in_dim, d_model)
        self.pos  = PositionalEncoding(d_model, max_len, dropout)
        layer     = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*2,
                                               dropout=dropout, batch_first=True)
        self.enc  = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, in_dim) — downsample FIRST, then encode
        if x.size(1) != self.target_seq_len:
            x = downsample_sequence(x, self.target_seq_len)  # → (B, 64, in_dim)
        x = self.proj(x)
        x = self.pos(x)
        return self.enc(x)   # (B, target_seq_len, d_model)


class EEGFeatureEncoder(nn.Module):
    """
    MLP encoder for EEG band-power features (63 channels × 5 bands = 315 dims).
    Projects to a compact representation suitable for concatenation with
    other trial-level features.
    """
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out_dim)


class TrialFeatureEncoder(nn.Module):
    """Encodes concatenated static features (eye stats + AU stats + shimmer stats + cursor + EEG)."""
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Emotion head (v2 with self-attention)
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedEmotionHead_v2(nn.Module):
    """
    Self-attention within the emotion head over (fused_token + personality_token),
    then MLP → per-task logits.
    """
    def __init__(self, fused_dim: int, persona_dim: int, n_tasks: int, n_classes: int,
                 hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        d = fused_dim + persona_dim
        self.norm  = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, num_heads=max(1, d // 64), batch_first=True,
                                           dropout=dropout)
        self.mlp   = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks * n_classes),
        )
        self.n_tasks   = n_tasks
        self.n_classes = n_classes

    def forward(self, fused: torch.Tensor, persona: torch.Tensor) -> torch.Tensor:
        # fused: (B, fused_dim),  persona: (B, persona_dim)
        x = torch.cat([fused, persona], dim=-1).unsqueeze(1)  # (B, 1, d)
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = x.squeeze(1)  # (B, d)
        out = self.mlp(x)  # (B, n_tasks*n_classes)
        return out.view(-1, self.n_tasks, self.n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class AFFECMultiTaskModel(nn.Module):
    """
    Transformer-Attention v2 multimodal multitask model.

    Inputs (per forward call):
        eye_seq      (B, T, eye_dim)
        pupil_seq    (B, T, pupil_dim)
        au_seq       (B, T, au_dim)
        gsr_seq      (B, T, gsr_dim)
        cursor_seq   (B, T, cur_dim)   ← NEW
        stim_emo     (B, 6)
        personality  (B, 5)
        eye_feats    (B, eye_feat_dim)
        au_feats     (B, au_feat_dim)
        shim_feats   (B, shim_feat_dim)
        eeg_feats    (B, eeg_dim)      ← NEW
        cursor_feats (B, cur_feat_dim) ← NEW

    Outputs:
        personality_pred   (B, 5)       continuous OCEAN predictions
        personality_delta  (B, 5)       trial-level personality delta
        emotion_logits     (B, n_tasks, n_classes)
        gender_logits      (B, 2) or None
    """

    def __init__(
        self,
        eye_dim:    int = 20,
        pupil_dim:  int = 4,
        au_dim:     int = 38,
        gsr_dim:    int = 8,
        cursor_dim: int = 5,

        eye_feat_dim:   int = 30,
        au_feat_dim:    int = 120,
        shim_feat_dim:  int = 35,
        cursor_feat_dim: int = 8,
        eeg_dim:        int = 315,

        d_model:   int = 128,
        n_heads:   int = 4,
        n_layers:  int = 2,
        n_tasks:   int = 4,     # felt_a, felt_v, perceived_a, perceived_v
        n_classes: int = 3,     # Low/Med/High
        dropout:   float = 0.1,

        use_gender:   bool = False,
        use_stim_emo: bool = True,
        use_eeg:      bool = True,
        use_cursor:   bool = True,
        modality_dropout_p: float = 0.1,  # probability to zero-out a modality token during training
    ):
        super().__init__()

        self.use_gender      = use_gender
        self.use_stim_emo    = use_stim_emo
        self.use_eeg         = use_eeg
        self.use_cursor      = use_cursor
        self.d_model         = d_model
        self.n_tasks         = n_tasks
        self.n_classes       = n_classes
        self.modality_dropout_p = modality_dropout_p

        # ── Sequence encoders ────────────────────────────────────────────
        self.eye_enc    = ModalityTransformerEncoder(eye_dim,    d_model, n_heads, n_layers, dropout)
        self.pupil_enc  = ModalityTransformerEncoder(pupil_dim,  d_model, n_heads, n_layers, dropout)
        self.au_enc     = ModalityTransformerEncoder(au_dim,     d_model, n_heads, n_layers, dropout)
        self.gsr_enc    = ModalityTransformerEncoder(gsr_dim,    d_model, n_heads, n_layers, dropout)
        if use_cursor:
            self.cursor_enc = ModalityTransformerEncoder(cursor_dim, d_model, n_heads, 1, dropout)

        # ── EEG encoder ──────────────────────────────────────────────────
        eeg_out = 64 if use_eeg and eeg_dim > 0 else 0
        if use_eeg and eeg_dim > 0:
            self.eeg_enc = EEGFeatureEncoder(eeg_dim, hidden=128, out_dim=eeg_out, dropout=dropout)

        # ── Fusion transformer (cross-modal) ─────────────────────────────
        # Note: norm_first removed so PyTorch can use nested-tensor optimisation.
        # dim_feedforward reduced to 2× (was 4×) — still sufficient for fusion.
        n_modalities = 4 + (1 if use_cursor else 0)
        fusion_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*2,
                                                   dropout=dropout, batch_first=True)
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=2)

        # Temporal pooling
        self.temporal_pool = TemporalAttentionPooling(d_model)

        # Task-specific attention queries
        self.persona_task_attn = TaskAttention(d_model, n_heads, n_queries=4)
        self.emotion_task_attn = TaskAttention(d_model, n_heads, n_queries=4)

        # ── Personality branch ───────────────────────────────────────────
        # Prediction path: pooled queries + stim_emo ONLY — true personality
        # is NOT included here so the model must predict from physiology.
        # True personality is injected separately (persona_cond_proj) into
        # the emotion head as auxiliary conditioning.
        stim_dim    = 6 if use_stim_emo else 0
        persona_in  = 4 * d_model + stim_dim          # no +5 for personality
        self.persona_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1), nn.GELU(),
            nn.Flatten(),
        )
        persona_conv_out = 8 * persona_in
        self.persona_fc  = nn.Sequential(
            nn.Linear(persona_conv_out, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
        )
        # Extra dropout before heads to regularise against overfitting training personalities
        self.persona_head_drop  = nn.Dropout(0.3)
        self.persona_head_base  = nn.Linear(128, 5)   # baseline OCEAN
        self.persona_head_delta = nn.Linear(128, 5)   # trial delta

        # Project true z-scored personality into 128-d space for emotion conditioning.
        # This keeps the emotion head's benefit from knowing ground-truth personality
        # while the prediction head is forced to work from physiological features only.
        self.persona_cond_proj = nn.Sequential(
            nn.Linear(5, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 128),
        )

        # ── Trial-level feature encoder ──────────────────────────────────
        total_feat = eye_feat_dim + au_feat_dim + shim_feat_dim + cursor_feat_dim + eeg_out
        self.trial_fc = TrialFeatureEncoder(total_feat, hidden=256, out_dim=128, dropout=dropout)

        # ── Emotion head v2 ──────────────────────────────────────────────
        # emotion input = pooled emotion queries + trial features
        emo_fused_dim = 4 * d_model + 128  # queries + trial_fc output
        self.emotion_head = AdvancedEmotionHead_v2(
            fused_dim=emo_fused_dim, persona_dim=128,
            n_tasks=n_tasks, n_classes=n_classes,
            hidden=256, dropout=dropout,
        )

        # ── Gender head ──────────────────────────────────────────────────
        if use_gender:
            self.gender_head = nn.Linear(128, 2)

        self.apply(init_weights)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _modality_dropout(self, token: torch.Tensor) -> torch.Tensor:
        """Zero out entire modality token with probability modality_dropout_p during training."""
        if self.training and self.modality_dropout_p > 0:
            mask = (torch.rand(token.size(0), 1, 1, device=token.device) > self.modality_dropout_p)
            token = token * mask.float()
        return token

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        eye_seq:      torch.Tensor,
        pupil_seq:    torch.Tensor,
        au_seq:       torch.Tensor,
        gsr_seq:      torch.Tensor,
        cursor_seq:   torch.Tensor,
        stim_emo:     torch.Tensor,
        personality:  torch.Tensor,
        eye_feats:    torch.Tensor,
        au_feats:     torch.Tensor,
        shim_feats:   torch.Tensor,
        eeg_feats:    torch.Tensor,
        cursor_feats: torch.Tensor,
    ):
        # ── Encode sequences ────────────────────────────────────────────
        eye_tok  = self._modality_dropout(self.eye_enc(eye_seq))      # (B, T, D)
        pup_tok  = self._modality_dropout(self.pupil_enc(pupil_seq))
        au_tok   = self._modality_dropout(self.au_enc(au_seq))
        gsr_tok  = self._modality_dropout(self.gsr_enc(gsr_seq))

        tokens = [eye_tok, pup_tok, au_tok, gsr_tok]
        if self.use_cursor:
            cur_tok = self._modality_dropout(self.cursor_enc(cursor_seq))
            tokens.append(cur_tok)

        # Encoders already output target_seq_len steps each — just concatenate
        fused_seq = torch.cat(tokens, dim=1)  # (B, n_mod * target_seq_len, D)

        fused_seq = self.fusion_transformer(fused_seq)   # (B, T_all, D)

        # ── EEG features ─────────────────────────────────────────────────
        if self.use_eeg and hasattr(self, "eeg_enc") and eeg_feats.size(-1) > 0:
            eeg_emb = self.eeg_enc(eeg_feats)  # (B, 64)
        else:
            eeg_emb = torch.zeros(eye_seq.size(0), 0, device=eye_seq.device)

        # ── Personality branch ───────────────────────────────────────────
        p_queries = self.persona_task_attn(fused_seq)   # (B, 4, D)
        p_flat    = p_queries.reshape(p_queries.size(0), -1)  # (B, 4*D)

        # Prediction path: physiological attention + stim ONLY (no ground-truth personality).
        # The model must infer personality from behaviour, not from the label itself.
        p_pred_parts = [p_flat]
        if self.use_stim_emo:
            p_pred_parts.append(stim_emo)
        p_pred_in = torch.cat(p_pred_parts, dim=-1)    # (B, 4D+stim)

        p_conv = self.persona_conv(p_pred_in.unsqueeze(1))  # (B, 8*p_pred_in)
        p_feat = self.persona_fc(p_conv)                     # (B, 128)

        # Predict OCEAN — targets are z-score normalised in the dataset
        p_drop        = self.persona_head_drop(p_feat)
        persona_base  = self.persona_head_base(p_drop)   # (B, 5) unbounded
        persona_delta = self.persona_head_delta(p_drop)  # (B, 5) unbounded
        persona_pred  = persona_base + persona_delta      # (B, 5)

        # For emotion / gender conditioning, ENRICH p_feat with the true z-scored
        # personality via a learned residual projection.  This lets the emotion head
        # benefit from knowing the participant's trait profile while keeping the
        # prediction head honest (it never sees its own target).
        p_cond      = self.persona_cond_proj(personality)  # (B, 128)
        p_feat_cond = p_feat + p_cond                       # (B, 128)

        # ── Trial-level features → emotion ──────────────────────────────
        feat_parts = [eye_feats, au_feats, shim_feats]
        if self.use_cursor and cursor_feats.size(-1) > 0:
            feat_parts.append(cursor_feats)
        if eeg_emb.size(-1) > 0:
            feat_parts.append(eeg_emb)

        # Pad shorter feature vectors with zeros to expected size
        trial_feat_raw = torch.cat(feat_parts, dim=-1)

        # Adapt linear input if sizes differ at runtime (first call)
        expected = self.trial_fc.net[0].in_features
        actual   = trial_feat_raw.size(-1)
        if actual < expected:
            trial_feat_raw = F.pad(trial_feat_raw, (0, expected - actual))
        elif actual > expected:
            trial_feat_raw = trial_feat_raw[:, :expected]

        trial_emb = self.trial_fc(trial_feat_raw)      # (B, 128)

        e_queries = self.emotion_task_attn(fused_seq)  # (B, 4, D)
        e_flat    = e_queries.reshape(e_queries.size(0), -1)  # (B, 4D)
        emo_fused = torch.cat([e_flat, trial_emb], dim=-1)    # (B, 4D+128)

        emotion_logits = self.emotion_head(emo_fused, p_feat_cond)  # (B, n_tasks, n_classes)

        # ── Gender head ──────────────────────────────────────────────────
        gender_logits = None
        if self.use_gender:
            gender_logits = self.gender_head(p_feat_cond)

        return persona_pred, persona_delta, emotion_logits, gender_logits
