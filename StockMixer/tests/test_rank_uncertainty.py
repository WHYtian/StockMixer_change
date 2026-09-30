"""Geometry, temporal boundaries and feature isolation for A1 diagnostics."""
import numpy as np
import pytest

from research.rank_uncertainty import (
    audit, deciles, expanding_folds, make_days, percentile, random_rank_loss,
    ridge_predict, select, summarise, within_decile_correlation,
)


def test_random_rank_geometry_matches_exact_enumeration_with_ties():
    p = percentile(np.array([1., 1., 2., 3., 3., 4.]))
    grid = (np.arange(p.size) + .5) / p.size
    exact = np.abs(p[:, None] - grid).mean(axis=1)
    np.testing.assert_allclose(random_rank_loss(p, p.size), exact)


def test_percentile_is_invariant_to_monotone_transform():
    x = np.array([-3., 1., 1., 4.])
    np.testing.assert_array_equal(percentile(x), percentile(np.exp(x)))


def test_geometry_does_not_require_predictive_information():
    n = 1000
    p = percentile(np.arange(n))
    baseline = random_rank_loss(p, n)
    assert baseline[0] > .49
    assert .249 < baseline[n // 2] < .251


@pytest.mark.parametrize("horizon", [1, 5, 10])
def test_temporal_folds_use_only_matured_labels(horizon):
    seen = []
    for fit, evaluation in expanding_folds(252, 96, 52, horizon):
        assert fit[-1] + horizon - 1 < evaluation[0]
        assert not set(fit) & set(evaluation)
        seen.extend(evaluation)
    assert seen == list(range(96, 252))


def test_rank_stratified_selection_preserves_decile_counts():
    p = percentile(np.arange(100))
    tie = np.random.default_rng(4).random(p.size)
    for risk in (p, -p, random_rank_loss(p, p.size)):
        take = select(risk, p, .6, True, tie)
        assert len(take) == 60
        assert np.bincount(deciles(p[take]), minlength=10).tolist() == [6] * 10


def test_constant_risk_within_deciles_has_no_residual_correlation():
    p = percentile(np.arange(100))
    assert np.isnan(within_decile_correlation(deciles(p), p, p))


def test_ridge_evaluation_does_not_change_fit_or_scaling():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(1000, 2))
    y = .3 + .1 * x[:, 0]
    z = rng.normal(size=(10, 2))
    prediction = ridge_predict(x, y, np.ones(1000), z)
    extra = np.vstack([z, [[1e9, -1e9]]])
    np.testing.assert_allclose(prediction, ridge_predict(x, y, np.ones(1000), extra)[:10])
    assert np.mean((prediction - (.3 + .1 * z[:, 0])) ** 2) < 1e-5


def synthetic_days(seed=6):
    rng = np.random.default_rng(seed)
    point = rng.normal(size=(3, 60, 18))
    mc = np.abs(rng.normal(size=point.shape))
    target = rng.normal(size=(60, 18))
    vol = rng.uniform(.01, .05, size=target.shape)
    eligible = np.ones(target.shape, dtype=bool)
    return point, mc, target, vol, eligible


def test_features_and_raw_risks_never_use_realised_targets():
    point, mc, target, vol, eligible = synthetic_days()
    original = make_days(point, mc, target, vol, eligible)
    changed = make_days(point, mc, -target, vol, eligible)
    for a, b in zip(original, changed):
        for key in a["features"]:
            np.testing.assert_array_equal(a["features"][key], b["features"][key])
        for key in a["risks"]:
            np.testing.assert_array_equal(a["risks"][key], b["risks"][key])


def test_crossfit_first_block_does_not_train_on_its_labels():
    point, mc, target, vol, eligible = synthetic_days()
    original = make_days(point, mc, target, vol, eligible)
    modified = target.copy()
    modified[:, 10:] *= -1
    changed = make_days(point, mc, modified, vol, eligible)
    rows, folds, risks = audit(original, initial=10, step=4)
    _, _, new_risks = audit(changed, initial=10, step=4)
    for a, b in zip(risks[:4], new_risks[:4]):
        for method in original[0]["features"]:
            np.testing.assert_allclose(a[method], b[method])
    assert len(rows) == 8 * 10
    assert len(folds) == 2
    summary = summarise(rows, draws=20, block=2)
    assert set(summary["paired_augmented_minus_control"]) == {"plus_mc", "plus_ensemble", "plus_rank", "plus_all"}
    assert "mse" not in summary["methods"]["raw_mc"]


def test_rank_disagreement_is_seed_rank_not_return_scale():
    point, mc, target, vol, eligible = synthetic_days()
    point[1] = 3 * point[0] + 5
    point[2] = 2 * point[0] - 9
    for day in make_days(point, mc, target, vol, eligible):
        np.testing.assert_allclose(day["risks"]["raw_rank"], 0, atol=1e-12)
        assert np.any(day["risks"]["raw_ensemble"] > 0)
