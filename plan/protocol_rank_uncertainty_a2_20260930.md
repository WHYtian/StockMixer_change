# A2: temporally separated ranking-reliability audit

Fixed before launching new training on 2026-09-30. This is a retrospective,
algorithmically out-of-sample audit, NOT an untouched-data or prospective
confirmation: these historical NASDAQ periods informed earlier research.

## Question and bounded budget

Does uncertainty add rank-error information beyond forecast score, forecast
percentile and historical volatility, without upstream checkpoint-selection
leakage? No portfolio rules, loss changes, parameter search or B-group training.
Two base-model windows, five fixed seeds each (20260813--20260817): ten runs.
Maximum 300 epochs each, early stopping patience 30, minimum training 30 epochs.
No extension or choice of winning method based on the new outcomes.

## Time boundaries (zero-based target-day indices; half-open)

| Outer window | Base training | Checkpoint selection | Initial risk fitting | Evaluation |
|---|---|---|---|---|
| 0 | [16,504) | [504,630) | [630,756) | [756,1008) |
| 1 | [16,756) | [756,882) | [882,1008) | [1008,1245) |

Base models restart from scratch in each window. Selection uses only its listed
period. Freeze the selected weights before generating any initial-risk/evaluation
predictions. Evaluate the risk predictor in blocks of 63 days, updating with
all previously matured post-selection labels (last block may be shorter).
Horizon 1: every fit target index is strictly less than the first evaluation
target index. No scaling, risk fitting or checkpoint selection sees future labels.
The two evaluation periods do not overlap; pooled evaluation has 489 days.
Second-window fitting may legitimately use earlier evaluation-period history.

NASDAQ original local dataset: 1026 stocks, 1245 days, five features; lookback 16.
StockMixer dropout 0.1 in mixer only; raw return target; Adam lr 0.001;
original MSE + ranking loss with alpha 0.1. Select the epoch with highest trailing
5-epoch mean selection RankIC. Minimum epochs limits stopping, not eligibility
of checkpoints. MC 50 draws after selection, separately seeded from training.
Base score is the five-seed deterministic point-prediction mean for EVERY method.
No claim that MC averaging improves the base forecast is tested here.

## Predictors and controls

Reuse A1 target, eligibility, features and quadratic ridge (penalty 0.001,
equal-date weights, train-only scaling, predictions clipped to [0,1]). Target
is absolute full-cross-section percentile rank displacement. Input control:
forecast percentile, score, absolute score, log lagged 16-day volatility and
volatility percentile. Retain all four augmentations: MC standard deviation,
cross-seed return standard deviation, cross-seed rank standard deviation, all.
This is error-risk prediction, not a full DEUP reproduction and not an identified
decomposition of irreducible versus epistemic uncertainty.

For each augmentation add a capacity-matched placebo: jointly permute its extra
columns within each date and forecast-rank decile, leaving controls unchanged.
Use deterministic label-independent RNG seed 20261001. One permutation per
configuration/date, not a randomization-significance test. Publish all placebo
outcomes. Also retain geometry, volatility, and three raw uncertainty baselines.

## Endpoints and statistics

Primary: equal-day-mean rank-error-prediction MSE differences, each genuine
augmentation minus control and minus its matched placebo (eight contrasts).
Pair all methods on exactly the same dates/stocks. Report ordinary 95% intervals
and Bonferroni simultaneous-family intervals: per-contrast coverage 99.375%,
family size eight. Use 10,000 paired circular block-bootstrap replicates, block
length 10, RNG seed 20260930. Sample separately within the two outer windows,
then pool with observed day-count weights; never wrap across a window boundary.
These are approximate dependence-aware intervals, not a universal coverage
guarantee and not a correction for all historical research choices.

A candidate clears this audit only if both adjusted upper bounds (vs control
and vs placebo) are below zero AND both window point differences vs control
are below zero. Otherwise report insufficient evidence; do not change endpoints.
Report every contrast and window, including negative and failed runs.

Secondary/exploratory: risk/error Spearman, within-decile correlation, retained
error and retained RankIC at 60% retention, both decile-stratified and global.
Never substitute retained RankIC for the primary MSE or for full-universe RankIC.
Raw uncalibrated standard deviations are ordering baselines, not MSE predictions.

## Provenance, limitations, publication

Save source/protocol/data/checkpoint/prediction hashes, indices, configs, selected
epoch, histories and complete daily results in new non-overwriting directories.
Workers are one per previously checked idle GPU; abort failed workers, do not
silently skip seeds. Unit tests and synthetic smoke tests precede full training.
Publish code, tests, protocol, aggregate and daily metrics; never credentials,
weights, raw stock data or per-stock predictions.

Known limits: historically inspected data/configurations; fixed-universe
survivorship bias; unknown upstream preprocessing provenance; future-label
observability masks are retrospective evaluation, not an executable universe;
only two historical windows and one market; intervals conditional on these
trained seeds; one placebo draw and a limited-capacity error predictor.
Later genuinely unseen-market confirmation remains necessary for strong claims.
