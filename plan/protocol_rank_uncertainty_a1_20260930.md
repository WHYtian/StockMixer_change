# A1: ranking uncertainty incremental-information pilot

Status: protocol fixed before reading the pilot's outcome metrics. This is a
development diagnostic, NOT a preregistered independent market result.

## Scope and provenance

- Do not retrain StockMixer, alter losses, change trading rules, or read test
  prediction files in this pilot.
- Input: NASDAQ E3 `dropout0.1_seed20260813` through `20260817`, **valid**
  predictions only. Dropout 0.1 is a historically inspected configuration, not
  a new unbiased choice. No claim of fresh model selection is permitted.
- Each base checkpoint was selected using the entire validation period.
  Chronological cross-fitting of the error predictor does NOT undo this
  upstream selection dependence. A future confirmation needs outer time splits
  with base-model selection confined to earlier data.
- Fixed base score: average deterministic point prediction over all five seeds.
- Target: absolute difference between forecast and realised midpoint percentile
  ranks on the same daily eligible cross-section. No portfolio simulation.
- Horizon 1 only. Reject truncated or misaligned arrays and inconsistent seeds.
- Saved MC draws are unavailable; rank disagreement means **cross-seed point
  rank disagreement**, not within-model MC ranking uncertainty.

## Comparators fixed before execution

1. Finite-universe random-rank expected displacement, using forecast rank only.
2. Historical 16-day volatility rank (uncalibrated error-ordering diagnostic).
3. Raw MC standard deviation, ensemble return standard deviation, cross-seed
   rank standard deviation (all uncalibrated ordering diagnostics).
4. Error predictor using forecast percentile, forecast score, absolute score,
   log trailing volatility and its cross-sectional percentile.
5. Same predictor plus MC uncertainty (log standard deviation and percentile).
6. Same predictor plus ensemble return uncertainty (log std and percentile).
7. Same predictor plus cross-seed rank disagreement (std and percentile).
8. Same predictor plus all three uncertainty feature groups.

The environment lacks sklearn and LightGBM. Use NumPy quadratic-feature ridge
regression for a transparent first diagnostic, not as a claimed DEUP/LightGBM
replication. Expand input features to degree 2 including pair interactions;
standardise only on fit dates. Minimise equal-date-weighted MSE plus 0.001 times
the squared standardised coefficient norm; intercept unpenalised. Clip predicted
rank loss to [0, 1]. No parameter search or result-driven method removal.

## Time protocol and metrics

- 252 validation days, first 96 for error-predictor fitting; subsequent blocks
  of 52 days evaluated using expanding history (three evaluation blocks).
- Only matured labels enter error-predictor fits. Horizon-1 target dates strictly
  precede the first evaluation target date. No shuffled stock-row split.
- Recompute eligibility from saved masks, finite scores and lagged volatility;
  identical eligibility for all methods. Ranks use this fixed cross-section.
- Primary: paired daily error-prediction MSE difference, augmented minus control.
- Secondary: daily Spearman correlation with rank error, including demeaning
  within forecast-rank deciles; 60% retention error and retained-set RankIC.
- Main retention comparison selects 60% **within each forecast-rank decile**
  so it cannot win merely by excluding the score tails. Unrestricted retention
  is reported separately, not substituted as the main comparison.
- Error targets always retain their original full-cross-section ranks, even
  after selection. Retained-set RankIC is secondary and not full-universe alpha.
- 2,000 paired circular moving-block bootstrap draws over evaluation dates,
  block length 10. Report all four augmented-control contrasts. Individual
  95% intervals are exploratory, without a multiplicity-adjusted success claim.
- Evaluation unit is date, not individual stock-day; include fold-level results.

## Artifacts and stopping rule

Save protocol/config/input hashes, all daily metrics, summary, fold membership,
and cross-fitted risk predictions in a new non-overwriting output directory.
Stop after this fixed pilot and report all results, including negative ones.
Do not select a winning method for test evaluation during this pilot.
