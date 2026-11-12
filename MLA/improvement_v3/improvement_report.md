# Restaurant turnover — October 12 follow-up

**Student register number:** submission

**Student name:** submission

**Team name:** submission

## What the screenshot establishes

The October 12 screenshot shows one submission, RMSE **12,204,615.6438253**, and **rank 7**. This is the same score as the October 11 rank-3 submission. The rank changed because competing entries moved ahead. The screenshot does not show a submission result for either improved candidate.

The leader shown has RMSE **11,946,659.07**. The original submission needs a reduction of more than **257,956.57** (2.11% of its RMSE) to beat that score, assuming it remains unchanged. No new portal score or rank is claimed.

## New local comparison

All rows below use original INR RMSE and retain every validation target. The two improved models and the originally submitted model use identical folds for each comparison.

| Validation | Originally submitted | Previous improved (v2) | Current candidate | Gain vs v2 |
|---|---:|---:|---:|---:|
| Shuffled 5-fold seed 71 | 19,956,845.01 | 19,891,238.91 | 19,870,083.61 | 21,155.30 |
| Shuffled 5-fold seed 2026 | 19,886,765.22 | 19,836,015.51 | 19,823,207.78 | 12,807.73 |
| Shuffled 5-fold seed 303 | 19,969,915.32 | 19,906,444.77 | 19,887,896.10 | 18,548.67 |
| Shuffled 5-fold seed 909 | 19,954,342.36 | 19,888,178.38 | 19,867,479.32 | 20,699.06 |
| City groups (stress test) | 20,227,279.16 | 20,092,949.44 | 20,129,673.34 | -36,723.89 |
| Mean of four shuffled assignments | 19,941,966.98 | 19,880,469.39 | 19,862,166.70 | 18,302.69 |

The current candidate lowers average shuffled-fold RMSE by **79,800.28 (0.400%)** versus the originally submitted model, and by **18,302.69** versus v2.

**Limits and tradeoff:** the city-isolation check is **36,723.89 worse than v2**. The conditional 95% row-bootstrap interval for the gain versus v2 is **[-9,580.43, 46,660.36]**, which includes zero. The incremental gain is not decisive under that check. The candidate is supported by four positive random-split comparisons, but it is not a proven improvement on unseen cities or on the hidden test set.

These assignments reuse the same labeled data used during development. Seed 909 was reserved for this round’s split comparison, but its restaurants were not an untouched holdout throughout all previous work. Scores are robustness evidence, not an unbiased external-test or nested model-selection estimate. Bootstrap uncertainty is conditional on the fitted models and does not cover all selection or distribution uncertainty.

## What changed

- Evaluated 42 additional model settings: 14 CatBoost settings averaged over three seeds, 20 spline/linear/kernel settings, and 8 Random Forest settings, plus coarse blend comparisons.
- Fixed the candidate after seed-42 development selection: 50% of the previous ten-model ensemble plus 50% of a three-seed CatBoost model using Bernoulli row subsampling.
- New component: depth 5, 850 trees, learning rate 0.04, Huber delta INR 40 million, L2 10, subsample 0.7; seeds 1042, 2042 and 3042 for full fitting.
- Random Forest settings all worsened development blend RMSE. The best smooth-model addition improved it by only about 3,085, so it was excluded.
- All validation targets retain their actual original values. Training-only robust losses and capping do not redefine the metric.

## Execution and file checks

- Executed all 10 code cells of the new notebook with zero saved errors.
- The experimental scripts trained the new fold models. The local notebook transparently reused hash-verified actual fold predictions and recalculated the metrics; its input hashes, model-configuration hash and file hashes were checked.
- All final models were refitted inside the notebook. Those predictions reproduced the independently generated candidate within INR 0.01. In Colab, no local cache exists and the included code trains the validation models.
- Also ran the notebook's fresh-training path on the complete seed-909 five-fold assignment without any prediction cache. All three models' predictions matched the recorded validation arrays exactly (maximum difference INR 0.00).
- Verified target/ID exclusion, original-scale RMSE, fold coverage and separation, canonical-city group separation, prediction row permutation, unseen and missing features, and exact CSV round-trip.
- Checked 500 data rows plus one header, two exact column names, unique restaurant IDs in original test order, finite nonnegative predictions and no added index column. Deliberately invalid submissions were rejected.
- Reviewed both saved diagnostic plots. The notebook includes code, actual outputs and inferences.

## Files and next action

- Current CSV: `submission.csv` in the main MLA folder.
- Matching executed notebook: `submission.ipynb` in the main MLA folder.
- Previous improved files are preserved in `improvement_v2/` and `improvement_v3/previous_delivery_backup/`. The originally submitted files remain in `improvement_v2/rank3_backup/`.

Upload the current CSV under **assignment M.Tech. MLA Hackathon -Restaurant Turnover Prediction** and record the actual returned RMSE. A hidden-test score cannot be calculated locally because the test targets are not provided. Upload the final executed notebook under **LLJ_Oct_2026** when your model choice is final; the stated assignment rule permits only one notebook submission.

CSV row registration numbers identify restaurants. The student number is used in the filenames and notebook cover. The faculty’s portal-registration naming requirement is separate from the submission-file format.

**Current CSV SHA256:** `348a70dce82f58e486c87cdcc69c1a3efd07056c72c6eaf47da6d79213cf9943`
