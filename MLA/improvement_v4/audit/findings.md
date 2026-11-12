# Independent v4 audit and bounded support experiment

## Current scored artifact

The root CSV exactly matches the v3 CSV. SHA256:
`348a70dce82f58e486c87cdcc69c1a3efd07056c72c6eaf47da6d79213cf9943`.

It has 500 data rows, exactly `Registration Number` and `Annual Turnover`, and
restaurant IDs in exactly the test-file order. All predictions are finite and
nonnegative. There is no student registration number in the prediction row IDs.

The latest user-provided portal evidence associates the submitted v3 candidate
with RMSE 12,094,600.3216516 and rank 3. Its earlier submission scored
12,204,615.6438253. No new v4 candidate has an observed portal score.

## Correctness and evaluation findings

- Train `Endorsed By` is normalized to test `Endoresed By` before feature creation.
- Restaurant IDs and turnover are excluded from all model features.
- Dates parse explicitly as day-month-year. CatBoost categoricals are strings;
  numeric missingness is preserved. Other estimators fit preprocessing within
  the training fold.
- Every validation row keeps its original annual turnover. Huber and target caps
  alter training objectives only. Reported primary scores are ordinary RMSE in INR.
- V3 notebook and scripts agree on fixed model settings, weights and seed
  schedules. The cache-derived v3 validation predictions equal 50% v2 plus
  50% mean of the three seed predictions. V3 output matches full-refit arrays.
- Earlier blend selection used all labeled rows' development OOF results; later
  splits reusing those labels are robustness checks, not an independent external
  test or a nested assessment of the whole selection process.
- V3's existing report correctly discloses the city-group downside and the
  conditional bootstrap interval spanning zero.

No correctness bug explains the old leaderboard rank decline.

## Untested data hypotheses identified

Train contains missing values in Facebook popularity (99), Instagram popularity
(56), restaurant tier (49), and ambience (25). None of the test rows is missing
these four fields. The union contains 214 training rows; 3,279 training rows have
complete values. This feature-based support distinction is observable without
test labels.

The complete-support group has 30 targets above INR 100 million; the missingness
group has 3. Thus the support distinction does not remove most of the extreme
turnover tail. It is not a justification for dropping difficult validation rows.

Additional obvious city spelling variants remain after lowercasing, for example
Banglore, Gurgoan, Bhubneshwar, Hderabad, and Punr. These were sent to the feature
agent. Merging clearly distinct cities into a region would need separate testing.

Service and comfort distributions differ between train and test, already explored
through removal variants. Do not shift their test values to force distributions
to match without independent evidence of a measurement change.

## Bounded training-weight experiment

Two settings were predeclared: give training rows missing any of the four fields
weight 0.25 or 0.5; all other training rows retain weight 1. Only feature values
determine weights. All validation rows retain their original labels and equal
weight in the primary RMSE calculation.

Each setting used the v3 CatBoost configuration: depth 5, 850 trees, learning rate
0.04, Huber delta INR 40 million, L2 10, Bernoulli subsampling 0.7, and two model
seeds 1042/2042 plus the zero-based fold number. Development is shuffled five-fold
seed 42 only. Reserved v4 splits were not used.

The baseline is exactly 50% cached v2 seed-42 OOF plus 50% cached v3 three-seed
subsampled component OOF. Its RMSE is **19,839,027.54**. The two-seed weighted
component was blended into this fixed baseline at 25% and 50%.

| Missing-pattern training weight | New component blend | Full RMSE | Gain vs v3 | Complete-support diagnostic gain |
|---|---:|---:|---:|---:|
| 0.25 | 25% | 19,842,003.15 | -2,975.61 | -5,755.69 |
| 0.25 | 50% | 19,848,537.51 | -9,509.97 | -14,603.54 |
| 0.50 | 25% | 19,835,512.50 | 3,515.04 | 2,125.07 |
| 0.50 | 50% | 19,835,000.54 | 4,027.00 | 1,372.98 |

The complete-support diagnostic scores only the 3,279 rows with no missing values
in those fields. It is conditional diagnostic evidence, not a replacement metric,
leaderboard estimate, or reason to exclude the remaining rows from primary RMSE.

Recommendation: 0.25 training weight does not help. The 0.50 training weight offers
only a small development blend gain and should not displace a stronger candidate
without fixed further validation. Neither weighted component improves v3 when used
alone. This experiment does not establish a portal improvement.

Artifacts in this directory: `support_weights.py`, `support_specs.json`,
`support_scores.csv`, `support_run.log`, and two full-row OOF `.npz` files containing
both individual seed predictions and their mean. No other model/data files were
edited by this audit.
