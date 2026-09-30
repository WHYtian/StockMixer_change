"""Portfolio arithmetic of the backtest."""

import numpy as np

from research.backtest import portfolio_returns, rule_scores, top_k_weights


def test_top_k_weights_pick_the_highest_scores_among_candidates() -> None:
    score = np.array([[0.1], [0.9], [0.5], [0.7]])
    candidates = np.array([[True], [False], [True], [True]])
    weights = top_k_weights(score, candidates, 2)
    assert weights[:, 0].tolist() == [0.0, 0.0, 0.5, 0.5]


def test_costs_charge_every_unit_traded() -> None:
    weights = np.array([[1.0, 0.0], [0.0, 1.0]])
    returns = np.array([[0.01, 0.0], [0.0, 0.02]])
    net, traded = portfolio_returns(weights, returns, cost_bps=10.0)
    assert traded.tolist() == [1.0, 2.0]
    assert np.allclose(net, [0.01 - 0.001, 0.02 - 0.002])


def test_filter_rule_removes_the_most_uncertain_share() -> None:
    mu = np.arange(10, dtype=float)[:, None]
    sigma = np.arange(10, dtype=float)[::-1][:, None].copy()
    eligible = np.ones((10, 1), dtype=bool)
    _, allowed = rule_scores(mu, sigma, eligible, "filter", 0.2)
    assert allowed[:, 0].tolist() == [False, False] + [True] * 8


def test_penalty_rule_with_zero_weight_keeps_the_ranking_of_mu() -> None:
    generator = np.random.default_rng(0)
    mu, sigma = generator.normal(size=(20, 3)), generator.random((20, 3))
    score, _ = rule_scores(mu, sigma, np.ones((20, 3), dtype=bool), "penalty", 0.0)
    assert np.array_equal(np.argsort(score, axis=0), np.argsort(mu, axis=0))


def test_market_gate_uses_only_earlier_days() -> None:
    from research.backtest import GATE_LOOKBACK, market_gate

    history = np.linspace(1.0, 2.0, GATE_LOOKBACK)
    current = np.array([1.0, 5.0, 1.0, 1.0])
    allowed = market_gate(history, current)
    assert allowed.tolist() == [True, False, True, True]
    changed = current.copy()
    changed[3] = 100.0
    assert market_gate(history, changed)[:3].tolist() == allowed[:3].tolist()


def test_consensus_needs_enough_seeds_to_agree() -> None:
    from research.backtest import consensus_candidates

    base = np.arange(10, dtype=float)[:, None]
    points = np.stack([base, base, base, base, base[::-1].copy()])
    eligible = np.ones((10, 1), dtype=bool)
    assert np.flatnonzero(consensus_candidates(points, eligible)[:, 0]).tolist() == [8, 9]
    assert not consensus_candidates(np.stack([base, base, base, base[::-1].copy(), base[::-1].copy()]), eligible).any()


def test_buffered_weights_keep_a_holding_until_it_leaves_the_exit_rank() -> None:
    from research.backtest import buffered_weights

    score = np.array([[3.0, 1.5, 0.0], [2.0, 2.0, 2.0], [1.0, 3.0, 3.0], [0.0, 0.0, 1.0]])
    weights = buffered_weights(score, np.ones_like(score, dtype=bool), top_k=1, exit_rank=3)
    assert np.flatnonzero(weights[:, 0]).tolist() == [0]
    assert np.flatnonzero(weights[:, 1]).tolist() == [0]
    assert np.flatnonzero(weights[:, 2]).tolist() == [2]


def test_trailing_mean_score_uses_today_and_earlier_days_only() -> None:
    from research.backtest import trailing_mean_score

    score = np.array([[1.0, 3.0, 5.0, 7.0]])
    assert trailing_mean_score(score, 2).tolist() == [[1.0, 2.0, 4.0, 6.0]]


def test_inverse_uncertainty_weights_sum_to_one_and_respect_the_cap() -> None:
    from research.backtest import SIZING_CAP, inverse_uncertainty_weights

    holdings = np.zeros((6, 1))
    holdings[:5, 0] = 0.2
    sigma = np.array([[0.001], [0.02], [0.02], [0.02], [0.04], [0.01]])
    weights = inverse_uncertainty_weights(holdings, sigma, 1.0)
    assert np.isclose(weights.sum(), 1.0) and weights[5, 0] == 0
    assert weights.max() <= SIZING_CAP / 5 + 1e-9
    assert weights[1, 0] > weights[4, 0]


def test_zero_gamma_gives_equal_weights() -> None:
    from research.backtest import inverse_uncertainty_weights

    holdings = np.array([[0.5], [0.5], [0.0]])
    weights = inverse_uncertainty_weights(holdings, np.array([[0.01], [0.05], [0.02]]), 0.0)
    assert weights[:, 0].tolist() == [0.5, 0.5, 0.0]
