"""Index alignment and statistics used by the evaluation scripts."""

import numpy as np

from research.aggregate import newey_west_t
from research.data import EODData, batch_at
from research.run_experiment import split_offsets
from research.uncertainty_eval import block_indices, partial_rank_correlation, trailing_volatility

STOCKS, DATES, LOOKBACK = 6, 60, 16


def _data() -> EODData:
    generator = np.random.default_rng(0)
    prices = 1.0 + generator.random((STOCKS, DATES)).astype(np.float32)
    returns = np.zeros_like(prices)
    returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
    features = np.repeat(prices[:, :, None], 5, axis=2)
    return EODData(features=features, masks=np.ones_like(prices), returns=returns, prices=prices)


def test_test_column_j_is_target_index_valid_end_plus_j() -> None:
    data, valid_end = _data(), 40
    offsets = split_offsets(valid_end, DATES, LOOKBACK, 1, 0)
    assert len(offsets) == DATES - valid_end
    for column, offset in enumerate(offsets):
        features, _, price, target = batch_at(data, offset, LOOKBACK, 1)
        assert np.array_equal(target[:, 0], data.returns[:, valid_end + column])
        assert np.array_equal(price[:, 0], data.prices[:, valid_end + column - 1])
        assert np.array_equal(features[:, -1, 0], data.prices[:, valid_end + column - 1])


def test_splits_do_not_share_target_days() -> None:
    train = {offset + LOOKBACK for offset in split_offsets(0, 30, LOOKBACK, 1, 0)}
    valid = {offset + LOOKBACK for offset in split_offsets(30, 40, LOOKBACK, 1, 0)}
    test = {offset + LOOKBACK for offset in split_offsets(40, DATES, LOOKBACK, 1, 0)}
    assert not (train & valid or valid & test or train & test)


def test_trailing_volatility_ends_the_day_before_the_target() -> None:
    data = _data()
    volatility = trailing_volatility(data.returns, data.masks, 40, 3, 16)
    assert np.allclose(volatility[:, 0], data.returns[:, 24:40].std(axis=1))
    changed = data.returns.copy()
    changed[:, 40:] += 1.0
    assert np.allclose(trailing_volatility(changed, data.masks, 40, 1, 16), volatility[:, :1])


def test_trailing_volatility_drops_returns_next_to_missing_days() -> None:
    data = _data()
    masks = data.masks.copy()
    masks[0, 30] = 0.0
    volatility = trailing_volatility(data.returns, masks, 40, 1, 16)
    kept = np.delete(data.returns[0, 24:40], [30 - 24, 31 - 24])
    assert np.isclose(volatility[0, 0], kept.std())


def test_newey_west_with_zero_lags_matches_plain_t() -> None:
    series = np.random.default_rng(1).normal(0.1, 1.0, 200)
    plain = series.mean() / (series.std(ddof=0) / np.sqrt(series.size))
    assert np.isclose(newey_west_t(series, 0), plain)


def test_partial_rank_correlation_removes_a_common_driver() -> None:
    generator = np.random.default_rng(2)
    driver = generator.normal(size=2000)
    x, y = driver + generator.normal(size=2000), driver + generator.normal(size=2000)
    assert abs(partial_rank_correlation(x, y, driver)) < 0.06


def test_block_indices_are_contiguous_blocks() -> None:
    indices = block_indices(50, 10, 4, np.random.default_rng(3))
    assert indices.shape == (4, 50)
    assert np.all((np.diff(indices[:, :10], axis=1) % 50) == 1)


def _panel() -> EODData:
    generator = np.random.default_rng(4)
    close = np.cumprod(1.0 + generator.normal(0, 0.02, (STOCKS, DATES)), axis=1).astype(np.float32) * 50.0
    features = np.stack([close, close * 1.01, close * 0.99, close, 1e6 * (1.0 + generator.random((STOCKS, DATES)))], axis=2).astype(np.float32)
    returns = np.zeros_like(close)
    returns[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    member = np.ones((STOCKS, DATES), dtype=bool)
    return EODData(features=features, masks=np.ones_like(close), returns=returns, prices=close, member=member, window_normalised=True)


def test_window_normalised_batch_uses_no_future_prices() -> None:
    data = _panel()
    features, _, price, target = batch_at(data, 10, LOOKBACK, 1)
    assert np.allclose(features[:, -1, 3], 1.0) and np.allclose(price, 1.0)
    assert np.array_equal(target[:, 0], data.returns[:, 10 + LOOKBACK])
    later = data.features.copy()
    later[:, 10 + LOOKBACK :, :] *= 3.0
    changed = EODData(features=later, masks=data.masks, returns=data.returns, prices=data.prices, member=data.member, window_normalised=True)
    assert np.array_equal(batch_at(changed, 10, LOOKBACK, 1)[0], features)


def test_window_normalised_mask_uses_membership_on_the_decision_day() -> None:
    data = _panel()
    base_day = 10 + LOOKBACK - 1
    data.member[0, base_day] = False
    data.member[1, base_day + 1] = False
    data.features[2, 12, :] = np.nan
    _, mask, _, _ = batch_at(data, 10, LOOKBACK, 1)
    assert mask[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0, 1.0, 1.0]


def test_smoothed_score_is_a_trailing_mean_and_waits_for_a_full_window() -> None:
    from research.run_experiment import smoothed_score

    history = [{"valid_rank_ic": value, "valid_mse": value} for value in (0.5, 0.1, 0.2, 0.3)]
    assert smoothed_score(history[:2], "rank_ic", 3) == float("-inf")
    assert np.isclose(smoothed_score(history, "rank_ic", 3), 0.2)
    assert np.isclose(smoothed_score(history, "mse", 3), -0.2)
    assert np.isclose(smoothed_score(history, "rank_ic", 1), 0.3)


def test_multi_day_target_is_the_compounded_forward_return() -> None:
    data = _data()
    for horizon in (1, 5):
        _, _, price, target = batch_at(data, 10, LOOKBACK, horizon)
        base = 10 + LOOKBACK - 1
        expected = data.prices[:, base + horizon] / data.prices[:, base] - 1.0
        assert np.allclose(target[:, 0], expected, atol=1e-5)
        assert np.array_equal(price[:, 0], data.prices[:, base])


def test_multi_day_splits_leave_no_overlap_between_target_periods() -> None:
    horizon = 5
    train = split_offsets(0, 30, LOOKBACK, horizon, 0)
    valid = split_offsets(30, 45, LOOKBACK, horizon, 0)
    last_train_target_day = train[-1] + LOOKBACK - 1 + horizon
    first_valid_target_day = valid[0] + LOOKBACK
    assert last_train_target_day == 29 and first_valid_target_day == 30
    assert valid[-1] + LOOKBACK - 1 + horizon == 44


def test_multi_day_mask_drops_stocks_that_halt_inside_the_target_period() -> None:
    data = _data()
    data.masks[3, 10 + LOOKBACK + 2] = 0.0
    _, mask, _, _ = batch_at(data, 10, LOOKBACK, 5)
    assert mask[3, 0] == 0 and mask[:3, 0].min() == 1


def test_moving_average_features_use_no_future_closes() -> None:
    from research.data import moving_average_features

    close = np.cumsum(np.ones((2, 40), dtype=np.float32), axis=1)
    features = moving_average_features(close)
    assert features.shape == (2, 40, 5)
    assert np.isclose(features[0, 30, 0], close[0, 26:31].mean())
    assert np.isclose(features[0, 35, 3], close[0, 6:36].mean())
    changed = close.copy()
    changed[:, 31:] += 100.0
    assert np.allclose(moving_average_features(changed)[:, :31], features[:, :31], equal_nan=True)
