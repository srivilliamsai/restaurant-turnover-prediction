# Feature representation error analysis

Development predictions only: shuffled 5-fold seed42, all original turnover labels. Candidate = 50% existing v3 + 50% two-seed categorical-ratings CatBoost. No models were refitted and no new validation splits were used.

Overall RMSE: v3 **19,839,027.54**, candidate **19,829,850.77**, reduction **9,176.77**.

## Fold consistency

| Fold | v3 RMSE | Candidate RMSE | Reduction | >100m rows |
|---|---:|---:|---:|---:|
| 1 | 18,810,714.84 | 18,763,004.99 | 47,709.86 | 4 |
| 2 | 22,358,306.93 | 22,370,528.00 | -12,221.07 | 6 |
| 3 | 20,233,123.00 | 20,264,114.88 | -30,991.88 | 11 |
| 4 | 14,853,650.46 | 14,828,882.35 | 24,768.11 | 7 |
| 5 | 22,001,590.89 | 21,976,677.63 | 24,913.26 | 5 |

## Tail sensitivity

| Segment | Rows | Reduction in RMSE | Candidate squared-error share | Share of total squared-error reduction |
|---|---:|---:|---:|---:|
| all | 3493 | 9,176.77 | 100.00% | 100.00% |
| <=100m | 3460 | 10,516.89 | 34.48% | 66.99% |
| >100m | 33 | 38,508.32 | 65.52% | 33.01% |
| >200m | 8 | 82,719.98 | 42.83% | 28.23% |

The >200m segment overlaps >100m. Negative reduction indicates deterioration. Target groups are diagnostic only; none are excluded from selection.

## Target deciles

| Target interval | Rows | Reduction in RMSE | Candidate error share |
|---|---:|---:|---:|
| (3499999.999, 12000000.0] | 439 | 184,674.92 | 7.00% |
| (12000000.0, 18000000.0] | 453 | 124,202.72 | 3.74% |
| (18000000.0, 20000000.0] | 225 | 83,330.07 | 1.14% |
| (20000000.0, 24000000.0] | 326 | 36,770.63 | 1.21% |
| (24000000.0, 30000000.0] | 438 | -44,116.59 | 1.40% |
| (30000000.0, 32500000.0] | 371 | -56,542.33 | 1.12% |
| (32500000.0, 35000000.0] | 277 | -132,842.09 | 0.88% |
| (35000000.0, 40000000.0] | 319 | -93,637.77 | 1.55% |
| (40000000.0, 48000000.0] | 301 | -39,890.60 | 2.73% |
| (48000000.0, 400000000.0] | 344 | -27,815.23 | 79.24% |

## Uncertainty and prediction changes

Conditional paired case-bootstrap: 5,000 resamples, RMSE-reduction 95% interval **[-15,187.81, 33,946.20]**; positive fraction **76.8%**. These are fixed, selected development OOF predictions. The interval omits feature-selection optimism, model retraining variation, and test distribution shift; it is not a guarantee of generalization.

OOF prediction changes: mean -117,385.61; mean absolute 511,134.27; median absolute 412,507.53; 95th percentile absolute 1,299,185.07; maximum absolute 3,778,113.42.

## Categorical representation audit

| Feature | Train categories | Test categories | Unseen test categories (rows) | Unseen dev validation rows |
|---|---:|---:|---|---:|
| Resturant Tier | 3 | 2 | None (0) | 0 |
| Restaurant City Tier | 2 | 2 | None (0) | 0 |
| Restaurant Zomato Rating | 6 | 5 | None (0) | 0 |
| Order Wait Time | 10 | 10 | None (0) | 0 |
| Staff Responsivness | 8 | 6 | None (0) | 1 |
| Value for Money | 7 | 6 | None (0) | 1 |
| Hygiene Rating | 9 | 8 | None (0) | 2 |
| Food Rating | 6 | 6 | None (0) | 0 |
| Overall Restaurant Rating | 6 | 6 | None (0) | 0 |
| Ambience | 12 | 11 | None (0) | 0 |
| Lively | 11 | 11 | None (0) | 0 |
| Service | 11 | 11 | None (0) | 1 |
| Comfortablility | 11 | 11 | None (0) | 0 |
| Privacy | 11 | 11 | None (0) | 0 |
| Live Music Rating | 9 | 8 | None (0) | 0 |
| Comedy Gigs Rating | 7 | 7 | None (0) | 1 |
| Value Deals Rating | 8 | 6 | None (0) | 2 |
| Live Sports Rating | 6 | 6 | None (0) | 0 |

Formatting uses `Missing` for nulls and `f'{v:g}'` otherwise, so integer/float representations of the same observed rating map to the same category. The inspected train and test rating values have no numeric-to-string collisions. Social popularity remains numeric. Unseen categories are valid CatBoost inputs and use learned fallback statistics; they do not cause invalid predictions, but performance for unseen values is uncertain.

The conversion treats distance/order between rating values as something to learn through categorical statistics. Therefore it cannot smoothly extrapolate a rating trend beyond training support. The proposed blend retains 50% of v3, which contains numeric rating models. Independent checks are required before retaining this candidate.
