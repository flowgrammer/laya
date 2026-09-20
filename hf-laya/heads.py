"""Head-layer factories and utilities for ``LayaForTypedDecision``."""

from __future__ import annotations

import torch
import torch.nn as nn


def make_head(hidden: int, layers: int, dropout: float = 0.1) -> nn.TransformerEncoder | None:
    """Transformer stack that runs on top of the encoder hidden states."""
    if layers <= 0:
        return None
    heads = max(1, hidden // 64)
    block = nn.TransformerEncoderLayer(
        hidden, heads, 4 * hidden, dropout,
        batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(block, layers, enable_nested_tensor=False)


def make_scorer(hidden: int) -> nn.Sequential:
    """Map a per-marker hidden vector to a scalar option score."""
    return nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, 1),
    )


def make_act_head(hidden: int, n_act: int) -> nn.Sequential:
    """Score ``n_act`` candidate actions (e.g. answer / escalate)."""
    return nn.Sequential(
        nn.Linear(hidden + 4, 256),
        nn.GELU(),
        nn.Linear(256, n_act),
    )


def act_features(logits: torch.Tensor, marker_mask: torch.Tensor) -> torch.Tensor:
    """Four-scalar description of an option-distribution shape used by the action head."""
    probs = torch.softmax(logits.detach(), dim=-1)
    counts = marker_mask.sum(-1).clamp_min(2).float()
    entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1) / torch.log(counts)
    top2 = probs.topk(2, dim=-1).values
    return torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy, counts / 255.0], dim=-1)
