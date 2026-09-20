"""Per-primitive temperature scaling and calibration metrics for typed-decision distributions."""

from __future__ import annotations

import math

import numpy as np
import torch

_QTYPE_NAMES = ("choice", "score", "noul")


def calibrated_softmax(
    logits: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
    temperature: torch.Tensor,
    overrides: dict[str, float],
) -> torch.Tensor:
    """Row-wise softmax scaled by per-``(qtype, option-count)`` temperature."""
    counts = marker_mask.sum(-1)
    scales = temperature[qtype]
    if overrides:
        scales = scales.clone()
        for i, (qt, k) in enumerate(zip(qtype.tolist(), counts.tolist())):
            key = f"{_QTYPE_NAMES[qt]}:{option_bucket(k)}"
            if key in overrides:
                scales[i] = overrides[key]
    scaled = logits / scales.clamp_min(1e-3).unsqueeze(-1)
    return torch.softmax(scaled, dim=-1)


def option_bucket(k: int) -> str:
    """Bucket used for ``temperature_by_options`` keys."""
    if k <= 2:
        return "2"
    if k <= 5:
        return "3-5"
    if k <= 10:
        return "6-10"
    return "11+"


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across ``bins`` confidence buckets.

    ``conf[i]`` is the model's confidence for prediction ``i``; ``correct[i]`` is
    1.0 if that prediction was right, 0.0 otherwise. Returns the weighted mean
    absolute gap between per-bin confidence and per-bin accuracy.
    """
    if len(conf) == 0:
        return math.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(ece)
