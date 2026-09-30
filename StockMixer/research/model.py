"""StockMixer variant with dropout that obeys PyTorch train/eval semantics."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


# Return variance bounds: standard deviations from about 0.25% to 37% per day.
MIN_LOG_VARIANCE, MAX_LOG_VARIANCE = -12.0, -2.0
INITIAL_LOG_VARIANCE = -7.8  # exp(-7.8 / 2) is about 2%, a typical daily return spread


def split_output(output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(predicted_price, log_variance_or_None)`` for either model variant."""
    if isinstance(output, tuple):
        return output
    return output, None


class MixerBlock(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int, dropout: float):
        super().__init__()
        self.first = nn.Linear(dimension, hidden_dimension)
        self.second = nn.Linear(hidden_dimension, dimension)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.first(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.second(x)
        return F.dropout(x, p=self.dropout, training=self.training)


class TriU(nn.Module):
    def __init__(self, time_steps: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(index + 1, 1) for index in range(time_steps)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for index, layer in enumerate(self.layers):
            current = layer(x[:, :, : index + 1])
            outputs.append(F.dropout(current, p=self.dropout, training=self.training))
        return torch.cat(outputs, dim=-1)


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps: int, channels: int, dropout: float):
        super().__init__()
        self.norm_time = nn.LayerNorm([time_steps, channels])
        self.norm_channel = nn.LayerNorm([time_steps, channels])
        self.time_mixer = TriU(time_steps, dropout)
        self.channel_mixer = MixerBlock(channels, channels, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.time_mixer(self.norm_time(inputs).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm_channel(x + inputs)
        return x + self.channel_mixer(x)


class MultiTimeMixer(nn.Module):
    def __init__(self, time_steps: int, channels: int, scale_steps: int, dropout: float):
        super().__init__()
        self.primary = Mixer2dTriU(time_steps, channels, dropout)
        self.scale = Mixer2dTriU(scale_steps, channels, dropout)

    def forward(self, inputs: torch.Tensor, downsampled: torch.Tensor) -> torch.Tensor:
        return torch.cat([inputs, self.primary(inputs), self.scale(downsampled)], dim=1)


class StockMixerLayer(nn.Module):
    def __init__(self, stocks: int, hidden_dimension: int):
        super().__init__()
        self.norm = nn.LayerNorm(stocks)
        self.first = nn.Linear(stocks, hidden_dimension)
        self.second = nn.Linear(hidden_dimension, stocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.permute(1, 0))
        return self.second(F.hardswish(self.first(x))).permute(1, 0)


class StockMixerDropout(nn.Module):
    """Architecture-compatible StockMixer with correctly controlled Dropout."""
    def __init__(self, stocks: int, time_steps: int, channels: int, market: int, scale: int, dropout: float = 0.1, head_dropout: bool = True, heteroscedastic: bool = False):
        super().__init__()
        del scale  # Retained for compatibility with the original constructor.
        scale_steps = 8
        self.dropout = dropout
        # The heads emit a price level, so noise on their inputs is amplified in return space.
        self.head_dropout = dropout if head_dropout else 0.0
        self.conv = nn.Conv1d(channels, channels, kernel_size=2, stride=2)
        self.mixer = MultiTimeMixer(time_steps, channels, scale_steps, dropout)
        output_steps = time_steps * 2 + scale_steps
        self.channel_fc = nn.Linear(channels, 1)
        self.stock_mixer = StockMixerLayer(stocks, market)
        self.time_fc = nn.Linear(output_steps, 1)
        self.stock_fc = nn.Linear(output_steps, 1)
        # Optional second head: log-variance of the next-day return (aleatoric uncertainty).
        self.heteroscedastic = heteroscedastic
        if heteroscedastic:
            self.time_fc_variance = nn.Linear(output_steps, 1)
            self.stock_fc_variance = nn.Linear(output_steps, 1)
            nn.init.zeros_(self.time_fc_variance.weight)
            nn.init.zeros_(self.stock_fc_variance.weight)
            nn.init.constant_(self.time_fc_variance.bias, INITIAL_LOG_VARIANCE)
            nn.init.zeros_(self.stock_fc_variance.bias)

    def forward(self, inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        downsampled = self.conv(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        mixed = self.mixer(inputs, downsampled)
        mixed = F.dropout(mixed, p=self.head_dropout, training=self.training)
        features = self.channel_fc(mixed).squeeze(-1)
        stock_features = self.stock_mixer(features)
        point = self.time_fc(F.dropout(features, p=self.head_dropout, training=self.training))
        stock = self.stock_fc(F.dropout(stock_features, p=self.head_dropout, training=self.training))
        if not self.heteroscedastic:
            return point + stock
        log_variance = self.time_fc_variance(features) + self.stock_fc_variance(stock_features)
        return point + stock, log_variance.clamp(MIN_LOG_VARIANCE, MAX_LOG_VARIANCE)


def return_ratio(predicted_price: torch.Tensor, base_price: torch.Tensor) -> torch.Tensor:
    return (predicted_price - base_price) / base_price


def stockmixer_loss(predicted_price: torch.Tensor, target: torch.Tensor, base_price: torch.Tensor, mask: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = return_ratio(predicted_price, base_price)
    regression = F.mse_loss(prediction * mask, target * mask)
    pairwise_prediction = prediction - prediction.transpose(0, 1)
    pairwise_target = target.transpose(0, 1) - target
    pairwise_mask = mask @ mask.transpose(0, 1)
    ranking = F.relu(pairwise_prediction * pairwise_target * pairwise_mask).mean()
    return regression + alpha * ranking, regression, ranking, prediction


def heteroscedastic_loss(predicted_price: torch.Tensor, log_variance: torch.Tensor, target: torch.Tensor, base_price: torch.Tensor, mask: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """beta-NLL (Seitzer et al., ICLR 2022) on returns plus the StockMixer ranking loss.

    ``beta = 0`` is the Gaussian NLL; ``beta = 1`` gives the mean the gradients of an MSE.
    """
    prediction = return_ratio(predicted_price, base_price)
    variance = log_variance.exp()
    nll = 0.5 * ((target - prediction) ** 2 / variance + log_variance)
    weighted = nll * variance.detach() ** beta
    regression = (weighted * mask).sum() / mask.sum().clamp(min=1.0)
    pairwise_prediction = prediction - prediction.transpose(0, 1)
    pairwise_target = target.transpose(0, 1) - target
    pairwise_mask = mask @ mask.transpose(0, 1)
    ranking = F.relu(pairwise_prediction * pairwise_target * pairwise_mask).mean()
    return regression + alpha * ranking


def cross_sectional_zscore(target: torch.Tensor, mask: torch.Tensor, clip: float = 3.0) -> torch.Tensor:
    """Standardise one day's returns across tradable stocks; masked stocks get zero.

    Removes the market-wide move of the day, so the regression loss is spent on
    relative performance.  Ranks within the day are unchanged.
    """
    count = mask.sum().clamp(min=1.0)
    mean = (target * mask).sum() / count
    spread = (((target - mean) ** 2 * mask).sum() / count).sqrt().clamp(min=1e-8)
    return ((target - mean) / spread).clamp(-clip, clip) * mask
