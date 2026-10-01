"""Development-only calibration and probability-level fusion utilities."""

from __future__ import annotations

import re

import numpy as np
import torch
from torch import nn


def dialogue_calibration_mask(clip_ids: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    """Select whole dialogues deterministically for calibration without turn leakage."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    dialogue_ids = []
    for clip_id in clip_ids.astype(str):
        match = re.fullmatch(r"dia(\d+)_utt\d+", clip_id)
        if not match:
            raise ValueError(f"Invalid MELD clip id: {clip_id}")
        dialogue_ids.append(int(match.group(1)))
    unique = np.asarray(sorted(set(dialogue_ids)))
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    count = max(1, round(len(unique) * fraction))
    selected = set(unique[:count].tolist())
    return np.asarray([value in selected for value in dialogue_ids], dtype=bool)


def fuse_probabilities(text: np.ndarray, audio: np.ndarray, alpha: float) -> np.ndarray:
    if text.shape != audio.shape or text.ndim != 2:
        raise ValueError("Text and audio probabilities must have the same 2D shape")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    combined = alpha * text + (1 - alpha) * audio
    return combined / combined.sum(axis=1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = torch.from_numpy(logits.astype(np.float32)) / temperature
    return torch.softmax(values, dim=1).numpy()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive scalar on the reserved calibration partition."""
    if logits.ndim != 2 or len(logits) != len(labels) or not len(labels):
        raise ValueError("Calibration logits and labels must be non-empty and aligned")
    raw = torch.from_numpy(logits.astype(np.float32))
    targets = torch.from_numpy(labels.astype(np.int64))
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(raw / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0))
