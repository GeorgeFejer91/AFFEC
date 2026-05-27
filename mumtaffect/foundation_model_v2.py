"""
Self-supervised foundation model components for MUMT-v2.

The pretraining objectives are:
1. cross-modal reconstruction: hide one or more modalities and reconstruct
   them from the remaining modalities plus event context;
2. next-step prediction: predict the future sequence state at a configurable
   horizon, e.g. one 20 Hz step = 50 ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class FoundationOutput:
    reconstruction: torch.Tensor
    next_prediction: torch.Tensor
    hidden: torch.Tensor


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MUMTFoundationModel(nn.Module):
    """
    Transformer sequence model over full-run multimodal grids.

    Inputs:
        x: normalized masked signal values `(batch, time, features)`.
        observed_mask: boolean observed-value mask with the same shape.
    """

    def __init__(
        self,
        n_features: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_features = int(n_features)
        self.max_seq_len = int(max_seq_len)
        self.input_projection = nn.Linear(self.n_features * 2, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.reconstruction_head = nn.Linear(d_model, self.n_features)
        self.next_head = nn.Linear(d_model, self.n_features)

        self.apply(_init_weights)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor) -> FoundationOutput:
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, time, features).")
        if x.size(-1) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.size(-1)}.")
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_seq_len={self.max_seq_len}.")

        observed_float = observed_mask.to(dtype=x.dtype)
        model_input = torch.cat([torch.nan_to_num(x, nan=0.0), observed_float], dim=-1)
        hidden = self.input_projection(model_input)
        hidden = hidden + self.position_embedding[:, : x.size(1), :]
        padding_mask = ~observed_mask.any(dim=-1)
        if padding_mask.any():
            hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        else:
            hidden = self.encoder(hidden)
        hidden = self.norm(hidden)
        return FoundationOutput(
            reconstruction=self.reconstruction_head(hidden),
            next_prediction=self.next_head(hidden),
            hidden=hidden,
        )


def targetable_modalities(
    modality_slices: Mapping[str, Sequence[int]],
    requested: Optional[Sequence[str]] = None,
) -> list[str]:
    if requested:
        requested_set = {name.strip() for name in requested if name.strip()}
        return [name for name in modality_slices if name in requested_set and name != "event"]
    return [name for name in modality_slices if name != "event"]


def make_ssl_inputs(
    x: torch.Tensor,
    observed_mask: torch.Tensor,
    modality_slices: Mapping[str, Sequence[int]],
    mask_modality_p: float = 0.35,
    mask_time_p: float = 0.05,
    requested_modalities: Optional[Sequence[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create masked inputs and reconstruction targets.

    At least one non-event modality is selected per sample. Time masking is
    additive and can force the model to reconstruct short temporal gaps too.
    """

    target_mask = torch.zeros_like(observed_mask, dtype=torch.bool)
    candidate_modalities = targetable_modalities(modality_slices, requested_modalities)
    if not candidate_modalities:
        raise ValueError("No targetable modalities found. Check modality_slices/requested_modalities.")

    batch_size = x.size(0)
    device = x.device
    for batch_index in range(batch_size):
        available_modalities = []
        for modality in candidate_modalities:
            start, stop = modality_slices[modality]
            if observed_mask[batch_index, :, int(start):int(stop)].any():
                available_modalities.append(modality)
        if not available_modalities:
            target_mask[batch_index] |= observed_mask[batch_index]
            continue

        selected = []
        for modality in available_modalities:
            if torch.rand((), device=device).item() < mask_modality_p:
                selected.append(modality)
        if not selected:
            random_index = int(torch.randint(0, len(available_modalities), (1,), device=device).item())
            selected = [available_modalities[random_index]]
        for modality in selected:
            start, stop = modality_slices[modality]
            target_mask[batch_index, :, int(start):int(stop)] |= observed_mask[batch_index, :, int(start):int(stop)]

    if mask_time_p > 0:
        time_mask = torch.rand(x.size(0), x.size(1), 1, device=device) < mask_time_p
        target_mask |= time_mask & observed_mask

    masked_x = x.masked_fill(target_mask, 0.0)
    return masked_x, target_mask


def foundation_ssl_loss(
    outputs: FoundationOutput,
    target_x: torch.Tensor,
    target_mask: torch.Tensor,
    observed_mask: torch.Tensor,
    next_horizon: int = 1,
    recon_weight: float = 1.0,
    next_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    recon_mask = target_mask.to(dtype=target_x.dtype)
    recon_error = ((outputs.reconstruction - target_x) ** 2) * recon_mask
    recon_loss = recon_error.sum() / recon_mask.sum().clamp_min(1.0)

    if next_horizon > 0 and target_x.size(1) > next_horizon:
        future = target_x[:, next_horizon:, :]
        prediction = outputs.next_prediction[:, :-next_horizon, :]
        next_mask = observed_mask[:, next_horizon:, :].to(dtype=target_x.dtype)
        next_error = ((prediction - future) ** 2) * next_mask
        next_loss = next_error.sum() / next_mask.sum().clamp_min(1.0)
    else:
        next_loss = torch.zeros((), dtype=target_x.dtype, device=target_x.device)

    total = recon_weight * recon_loss + next_weight * next_loss
    return total, {"recon_loss": recon_loss.detach(), "next_loss": next_loss.detach()}
