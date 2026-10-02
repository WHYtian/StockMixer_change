# External A-share ranking-reliability audit

Frozen 2026-10-02. This is a retrospective external-market holdout audit, not
an untouched-data or prospective confirmation: the historical A-share test period
was inspected in the earlier project.

- Dataset: point-in-time CSI300+CSI500 panel, 1,579 stocks × 2,853 dates.
- Base models: existing five `dropout=0.1`, Mixer-only, raw-label runs with
  training indices through 2188 and validation indices 2189–2430 (calendar 2024).
- Test/evaluation: indices 2431–2852 (2025-01 through 2026-09), 422 dates.
- No validation predictions are opened; only frozen `test_predictions.npz` files.
- All five seeds share the same deterministic score and test universe; no model,
  feature, threshold or method is selected from this test period.
- Controls, four uncertainty augmentations, within-rank-decile placebo features,
  quadratic ridge error model, 10-day block bootstrap and primary endpoint follow
  A2. With one external window, there is no cross-window consistency criterion.
- Primary comparisons are the four genuine augmentations versus control and their
  matched placebo. The same 10,000 bootstrap draws and Bonferroni family of eight
  comparisons are used. RankIC after 60% retention is secondary only.

The analysis tests transfer of a frozen A2-style reliability signal to A shares;
it does not establish trading performance, causal uncertainty, or a universal
market effect. It must not be presented as an untouched prospective result.
