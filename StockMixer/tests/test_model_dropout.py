"""Dropout placement and mode control of the research StockMixer."""

import torch

from research.model import StockMixerDropout

STOCKS, STEPS, CHANNELS = 12, 16, 5


def _model(dropout: float, head_dropout: bool) -> StockMixerDropout:
    torch.manual_seed(0)
    return StockMixerDropout(STOCKS, STEPS, CHANNELS, 20, 3, dropout, head_dropout=head_dropout)


def _inputs() -> torch.Tensor:
    # A private generator keeps the global RNG, which drives dropout masks, untouched.
    return torch.rand(STOCKS, STEPS, CHANNELS, generator=torch.Generator().manual_seed(1))


def test_eval_mode_is_deterministic() -> None:
    model = _model(0.5, head_dropout=True).eval()
    assert torch.equal(model(_inputs()), model(_inputs()))


def test_train_mode_is_stochastic() -> None:
    model = _model(0.5, head_dropout=False).train()
    assert not torch.equal(model(_inputs()), model(_inputs()))


def test_no_dropout_outside_mixer_when_head_dropout_is_off() -> None:
    model = _model(0.5, head_dropout=False).train()
    model.mixer.eval()
    assert torch.equal(model(_inputs()), model(_inputs()))


def test_head_dropout_is_active_when_requested() -> None:
    model = _model(0.5, head_dropout=True).train()
    model.mixer.eval()
    assert not torch.equal(model(_inputs()), model(_inputs()))


def test_heteroscedastic_model_returns_a_bounded_log_variance() -> None:
    from research.model import MAX_LOG_VARIANCE, MIN_LOG_VARIANCE, heteroscedastic_loss

    torch.manual_seed(0)
    model = StockMixerDropout(STOCKS, STEPS, CHANNELS, 20, 3, 0.1, head_dropout=False, heteroscedastic=True)
    price, log_variance = model(_inputs())
    assert price.shape == log_variance.shape == (STOCKS, 1)
    assert log_variance.min() >= MIN_LOG_VARIANCE and log_variance.max() <= MAX_LOG_VARIANCE
    base = torch.ones(STOCKS, 1)
    loss = heteroscedastic_loss(price, log_variance, torch.zeros(STOCKS, 1), base, torch.ones(STOCKS, 1), 0.1, 0.5)
    loss.backward()
    assert torch.isfinite(loss) and model.time_fc_variance.bias.grad is not None


def test_beta_one_gives_the_mean_the_gradient_of_a_halved_squared_error() -> None:
    from research.model import heteroscedastic_loss

    price = torch.tensor([[1.02], [0.97]], requires_grad=True)
    log_variance = torch.tensor([[-6.0], [-8.0]])
    target, base, mask = torch.tensor([[0.01], [0.00]]), torch.ones(2, 1), torch.ones(2, 1)
    heteroscedastic_loss(price, log_variance, target, base, mask, 0.0, 1.0).backward()
    expected = -(target - (price.detach() - base)) / 2.0
    assert torch.allclose(price.grad, expected, atol=1e-7)


def test_cross_sectional_zscore_keeps_ranks_and_ignores_masked_stocks() -> None:
    from research.model import cross_sectional_zscore

    target = torch.tensor([[0.03], [-0.01], [0.50], [0.01], [0.02]])
    mask = torch.tensor([[1.0], [1.0], [0.0], [1.0], [1.0]])
    scored = cross_sectional_zscore(target, mask)
    kept = mask[:, 0] > 0
    assert scored[2, 0] == 0
    assert torch.equal(scored[kept, 0].argsort(), target[kept, 0].argsort())
    assert abs(float(scored[kept].mean())) < 1e-6
    assert abs(float(scored[kept].pow(2).mean()) - 1.0) < 1e-5
    shifted = cross_sectional_zscore(target + 0.05, mask)
    assert torch.allclose(shifted, scored, atol=1e-5)
