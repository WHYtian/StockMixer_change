"""MC Dropout inference with price-to-return conversion per sample."""

from __future__ import annotations

import torch
from torch import nn

from .model import return_ratio, split_output


def predict_mc_returns(model: nn.Module, features: torch.Tensor, base_price: torch.Tensor, samples: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean, std, and all return samples, shaped [stocks, 1] / [T, stocks, 1]."""
    if samples < 2:
        raise ValueError("MC Dropout requires at least two samples")
    was_training = model.training
    model.train()  # This model has no BatchNorm; Dropout is the only mode-sensitive layer.
    try:
        with torch.no_grad():
            returns = torch.stack([return_ratio(split_output(model(features))[0], base_price) for _ in range(samples)])
    finally:
        model.train(was_training)
    return returns.mean(dim=0), returns.std(dim=0, unbiased=True), returns
