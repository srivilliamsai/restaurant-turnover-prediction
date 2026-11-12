# Superseded four-setting V10 draft

**This historical draft was replaced by the user-approved six-setting V10 contract. See the [implemented plan](../improvement_v10/PLAN.md).**

**Preserve V9: portal RMSE 11,904,515.9864996; rank 9 in the supplied October 27 screenshot.**

Status: proposed after review of 29 supplied Python files. Only source inspection and data diagnostics are complete. No V10 model training or portal upload has started.

## 1. Hypothesis and scope

The next hypothesis is that a small number of structured, target-free features can improve the successful TabPFN component. Some features were tried previously in CatBoost or Ridge, but the selected V9 TabPFN uses the V7-clean 40-column representation. The experiment tests their value for that particular model; it does not assume more columns improve accuracy.

Use the four official datasets, all 3,493 original training targets and the official 500 test rows. Exclude restaurant IDs and targets from every feature builder. Do not add another model-family, target-transformation or parameter search. No target filtering, target rounding, pseudo-labels, external answer files or global fitted calibrator.

The development baseline is the actual V9 ensemble, not V7. Reuse existing OOF predictions only after checking input, source, dependency/platform contract, model configuration, row order, fold assignments and prediction hashes. V7 refits must respect the documented Mac/Colab reproduction limitation. Do not silently replace its verified predictions with a different Linux fit.

## 2. Freeze four representations

Use TabPFN v2 regression, raw target in millions of INR, 16 estimators, predictive mean, seed 10242, and the working V9 CUDA/float32/low-memory contract. Freeze the existing dependency pins and official checkpoint SHA256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`. Learn category vocabularies inside the current training subset; declare categorical columns explicitly and map unseen values to missing.

| Setting | Features supplied to the new TabPFN component | Purpose |
|---|---|---|
| C | V7 features plus the cleanup block | Isolate semantic/category cleanup. |
| S | V7 features plus the 16 structured features below | Isolate structured representation. |
| C+S | Both blocks | Check their combination. |
| C+S+G | Both blocks plus eight training-subset group features | Check whether limited relative/context features help. |

### Cleanup block C

- Keep the existing V7 `City` column. Append one revised canonical-city view using the exact conservative literal map recorded in `source_and_data_audit.json`. Missing/unknown canonical cities are missing values. Preserve V7’s unknown-city indicator. No fuzzy matches, blanket punctuation deletion or additional metropolitan pooling.
- In the new component only, replace Zomato zero with missing and append a zero-value indicator. This is a tested interpretation of the 13 values outside the documented range; it is not a proven correction. The frozen V7 and V9 control remain unchanged.

The measured map reduces unseen test-city rows from 24 to 20. That limits the direct coverage benefit; it is not a basis for a large score forecast.

### Structured block S: exactly 16 additional features

Compute row-level features before any model preprocessing. Use available values for means; preserve missing results when there are no available inputs. Standard deviations use `ddof=1`, leaving fewer than two observations missing. Products/differences remain missing when an operand is missing.

1. Mean normalized quality: staff/8, value-for-money/7, hygiene/10, food/10, Zomato/5, overall/10.
2. Standard deviation of those six normalized quality values.
3. Hygiene/10 + staff/8 + value-for-money/7; require all three values.
4. Sum of the five binary facility fields; use `min_count=1` so an entirely unavailable row does not become zero.
5. Number of offered event types, using the four existing nonmissing event indicators.
6. Mean normalized offered-event rating: music/10, comedy/6, value-deals/7, sports/6. The dictionary defines blank event ratings as not offered.
7. Maximum of those normalized event ratings; leave missing if none are offered.
8. Mean of Ambience, Lively, Service, Comfortablility and Privacy.
9. Standard deviation of those five experience ratings.
10. Mean Facebook/Instagram popularity, using available values.
11. Facebook popularity × Instagram popularity.
12. Absolute Facebook/Instagram popularity difference.
13. Restaurant age in years at fixed reference 2026-01-01, bounded below by zero; missing dates remain missing.
14. Restaurant age × hygiene/10.
15. Restaurant age × social mean.
16. Restaurant tier × city tier, preserving unknown tiers as missing.

When C is also enabled, derive quality features after its Zomato treatment. Do not add six redundant individually rescaled quality columns, arbitrary aptitude weights, personality reinterpretations or extra date interactions during this round.

### Group block G: exactly eight additional features

Fit these maps using the current training subset only. No validation/test rows contribute to counts, sums, category pooling or global means.

- `log1p` row counts for revised canonical city and normalized full cuisine: two features.
- For each of those two groups, deviations from the group mean for social mean, hygiene/10 and age: six features.

Shrink each group mean toward its training-subset global mean with fixed strength 20: `(group_sum + 20 * global_mean) / (group_nonmissing_count + 20)`. For an unseen group, the count is zero and the mean is the training global mean. Preserve missing numeric inputs. Store the learned maps with the model and use them unchanged at inference. Do not search smoothing strengths or add group z-scores.

## 3. Compare a fixed ensemble structure

Let `P7` be the verified V7 predictions, `P9` the selected 16-estimator TabPFN predictions, and `Pnew` one of the four new components.

`V9 = 0.5 * P7 + 0.5 * P9`

`V10(w) = 0.5 * P7 + 0.5 * ((1-w) * P9 + w * Pnew)`

Test `w = 0.25, 0.50, 0.75, 1.00`. This keeps the proven V7 share fixed while measuring replacement of the TabPFN side. There are **four new fitted settings and 16 selectable blend configurations**, plus unchanged V9. Report component-alone RMSE as a diagnostic; component-alone models are not additional selectable configurations in this round.

No refinement grid or extra estimators are added after screening. Save checkpoints after each completed outer fold. Cache identity must include feature source/map hashes, dependencies, checkpoint, training-row indices, model settings, folds and output-prediction hash. Reject stale, incomplete or corrupt checkpoints.

## 4. Development, freeze and confirmation

1. Screen on identical five-fold assignments **42 and 71**, with every original validation target.
2. A selectable blend must beat V9 on both assignments. Rank by mean full-target RMSE, then smaller replacement weight, then fewer added features, then setting name.
3. Advance **at most one** finalist. Freeze its representation, maps/procedures, parameters, execution contract and replacement weight before confirmation. If no blend improves both assignments, stop and retain V9.
4. Confirm against V9 on random five-fold assignments **48211, 48212 and 48213**, and target-stratified assignment **27421**. These labels and assignments have been examined before; they are robustness checks, not an untouched external test.
5. Run normalized-city grouped validation using the frozen conservative revised city map. Both V9 and the finalist must use exactly the same grouped folds. If the revised city map changes the previous fold membership, refit both baselines and candidate on those new folds; old city predictions cannot be reused under a new split hash. Unknown cities stay in a common group.
6. Qualify only if the candidate improves full-target RMSE on **all three random assignments**, with no more than **1% deterioration** on either stratified or city-grouped validation. If it fails, report the failure and keep V9; do not retune on confirmation labels.

Report fold RMSE, all failed configurations, and turnover groups ≤40M, 40–80M and >80M. For each check, show the ten largest positive row contributions and the extreme-turnover group’s share of net squared-error reduction. These target-dependent diagnostics must not determine inference rules.

Use 5,000 paired-row bootstrap resamples, drawing the same restaurant indices across the three random assignments. Report a conditional interval for the mean RMSE improvement; it excludes model-search, retraining and hidden-test distribution uncertainty.

## 5. Compute and delivery

Use the existing free Colab T4 route when available, with persistent checkpoints and a bounded pilot at the actual outer-fold training size. Measure pilot time and peak memory before committing to the full run. The proposed target is roughly **8–12 hours**, with an absolute **18-hour cap**, subject to observed speed and free GPU availability. These are planning allowances, not a running-job estimate or a guaranteed Colab session duration. No paid resources are assumed. Finish earlier when the declared comparisons are complete and reserve at least two hours for reproduction/delivery.

Build an isolated `improvement_v10` workflow with explicit data/output directories and audit, pilot, screen, confirm and export stages. Do not change the V9 environment, scored CSV or matching notebook. Aim to finish artifacts by October 28; the portal’s October 29 deadline timezone still needs confirmation.

Before export, verify:

- Original input hashes, target/ID exclusion, fold isolation and rejection of stale/corrupt checkpoints.
- Missing numeric values, zero-Zomato interpretation, unseen cities/categories and all-missing event rows.
- Row permutation and prediction-batch invariance, including predicting rows individually versus together.
- Exactly 500 rows, original unique restaurant IDs/order, headers `Registration Number`, `Annual Turnover`, and finite nonnegative predictions.
- Separate reproduction of the preserved V9 baseline and a fresh standalone execution of the selected V10 notebook within **INR 0.01** of their exported predictions, under the pinned execution contract. Carry forward V9’s documented verified V7-state approach rather than claiming an identical fresh Linux V7 fit.

Deliver a qualified CSV with its matching freshly executed notebook, a report of measured comparisons including failures, dependency/checkpoint/input/source manifests and submission hashes. If nothing qualifies, deliver the report and continue recommending the byte-identical scored V9.

Use one portal trial for the qualified finalist. Record its returned RMSE against the exact CSV hash and retain the lowest **observed** portal score. A local validation gain does not guarantee a portal improvement. Keep the one-time coding upload pending the final model choice.

## 6. What this plan can establish

It can test whether these specific feature ideas improve V9 under the declared checks. It cannot establish how another participant achieved 4,796,740.56 or promise to beat that score. The submitted code contains no reproducible evidence of such a result. The immediate objective is a measured improvement over **11,904,515.99** while preserving the existing best.
