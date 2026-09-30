"""Temporal isolation, capacity controls and family-level reporting for A2."""
import copy

import numpy as np
import pytest

from research.rank_a2 import (
    AUGMENTATIONS, SEEDS, WINDOWS, add_placebos, boundaries, interval,
    stratified_blocks, summarise_a2, validate_config,
)
from research.rank_uncertainty import audit, deciles, make_days


@pytest.mark.parametrize("window", WINDOWS)
def test_all_base_selection_targets_precede_risk_fit(window):
    offsets = boundaries(window)
    train = np.asarray(offsets["train"]) + 16
    select = np.asarray(offsets["selection"]) + 16
    later = np.asarray(offsets["post_selection"]) + 16
    assert train.max() < select.min()
    assert select.max() < later.min()
    assert later[0] == window[1] and later[-1] == window[3]-1
    assert window[2]-window[1] == 126


def test_registered_evaluation_windows_are_disjoint_489_days():
    dates = [i for window in WINDOWS for i in range(window[2], window[3])]
    assert len(dates) == len(set(dates)) == 489


def test_reject_invalid_boundaries():
    with pytest.raises(ValueError):
        boundaries((504, 756, 630, 1008))


def synthetic_days():
    rng = np.random.default_rng(17)
    point = rng.normal(size=(5, 60, 18))
    target = rng.normal(size=(60, 18))
    return make_days(point, np.abs(rng.normal(size=point.shape)), target,
                     rng.uniform(.01, .05, target.shape), np.ones(target.shape, bool))


def test_placebos_preserve_controls_and_within_decile_joint_extra_rows():
    days = synthetic_days()
    original = copy.deepcopy(days)
    add_placebos(days)
    for day, before in zip(days, original):
        groups = deciles(day["p"])
        for method in AUGMENTATIONS:
            real = day["features"][method]
            fake = day["features"][method.replace("plus_", "placebo_")]
            np.testing.assert_array_equal(real, before["features"][method])
            np.testing.assert_array_equal(fake[:, :5], real[:, :5])
            for bucket in np.unique(groups):
                assert sorted(map(tuple, fake[groups == bucket, 5:])) == sorted(map(tuple, real[groups == bucket, 5:]))


def test_placebos_are_reproducible_and_do_not_read_labels():
    a, b = synthetic_days(), synthetic_days()
    for day in b:
        day["loss"][:] = 0
        day["truth"][:] = 999
    add_placebos(a)
    add_placebos(b)
    for left, right in zip(a, b):
        for key in left["features"]:
            np.testing.assert_array_equal(left["features"][key], right["features"][key])


def test_block_bootstrap_never_crosses_outer_windows():
    labels = np.r_[np.zeros(31, int), np.ones(23, int)]
    indices = stratified_blocks(labels, draws=100, block=10)
    assert indices.shape == (100, 54)
    assert (indices[:, :31] < 31).all()
    assert (indices[:, 31:] >= 31).all()
    np.testing.assert_array_equal(indices, stratified_blocks(labels, draws=100, block=10))


def test_adjustment_widens_intervals_and_nan_is_not_success():
    x = np.arange(54, dtype=float)
    indices = stratified_blocks(np.r_[np.zeros(31), np.ones(23)], draws=100)
    regular, adjusted = interval(x, indices), interval(x, indices, .05/8)
    assert adjusted["lower"] <= regular["lower"] <= regular["upper"] <= adjusted["upper"]
    assert interval(np.full(54, np.nan), indices) is None


def test_config_rejects_smoke_wrong_dates_and_wrong_seed():
    config = dict(window=0, seed=SEEDS[0], boundaries=list(WINDOWS[0]), smoke=False,
                  dropout=.1, dropout_sites="mixer", horizon=1, lookback=16, label="raw",
                  learning_rate=.001, alpha=.1, mc_samples=50, epochs=300, patience=30,
                  min_epochs=30, selection_metric="rank_ic", selection_smoothing=5,
                  protocol_sha256="frozen")
    validate_config(config, 0, SEEDS[0], "frozen")
    for key, value in (("smoke", True), ("boundaries", [504, 756, 882, 1008]), ("seed", 1)):
        with pytest.raises(ValueError):
            validate_config({**config, key: value}, 0, SEEDS[0], "frozen")


def test_audit_all_placebos_temporal_isolation_and_eight_contrasts():
    days = synthetic_days()
    add_placebos(days)
    changed = copy.deepcopy(days)
    for day in changed[10:]:
        day["loss"] = 1-day["loss"]
    rows, _, saved = audit(days, initial=10, step=4)
    _, _, new_saved = audit(changed, initial=10, step=4)
    for old, new in zip(saved[:4], new_saved[:4]):
        for method in days[0]["features"]:
            np.testing.assert_allclose(old[method], new[method])
    for row in rows:
        row["outer_window"] = int(row["day"] >= 14)
        row["target_day"] = row["day"]
    summary = summarise_a2(rows, draws=100)
    assert len(summary["contrasts"]) == 8
    assert len(summary["methods"]) == 14
    assert summary["evaluation_days"] == 8
    assert summary["methods"]["raw_mc"]["mse"] is None
    with pytest.raises(ValueError):
        summarise_a2(rows + rows, draws=10)


def test_success_requires_both_references_and_both_windows():
    rows = []
    methods = ["control", *AUGMENTATIONS, *[m.replace("plus_", "placebo_") for m in AUGMENTATIONS]]
    metrics = ("rho", "rho_within_decile", "loss60_stratified", "rankic60_stratified", "loss60_global", "rankic60_global", "full_rank_ic")
    for day in range(40):
        for method in methods:
            mse = .05
            if method == "plus_mc":
                mse -= .01  # Pass both references, both windows.
            if method in ("plus_rank", "placebo_rank"):
                mse -= .02  # No incremental gain over matched placebo.
            if method == "plus_ensemble" and day < 20:
                mse -= .03  # No negative difference in window 1.
            rows.append(dict(target_day=day, outer_window=int(day >= 20), method=method,
                             mse=mse, **{m: .1 for m in metrics}))
    assert summarise_a2(rows, draws=100)["candidates_passing"] == ["plus_mc"]
