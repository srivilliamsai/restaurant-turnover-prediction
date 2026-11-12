# Review of the 29 supplied Gemini Python files

Review date: October 27, 2026. Scope: static inspection of every supplied file, syntax parsing, comparison with the existing V2–V9 implementations, and read-only checks on the four official datasets. No attached script was executed and no model was trained during this review.

**Recommendation: preserve scored V9 and test a small set of new feature representations for its successful TabPFN component. The attachments provide hypotheses, not measured evidence of a better submission.**

Current confirmed portal result: **11,904,515.9864996**, rank **9** in the supplied screenshot. V9 is 50% V7 plus 50% raw-target TabPFN v2 with V7 features and 16 estimators. Its mean RMSE over the three random confirmation assignments was **19,757,137.52**. Validation RMSE and portal RMSE are different measurements; their ratio is not a reliable score forecasting rule.

## Coverage and execution status

- 29 supplied files; 27 unique file contents. Two pairs are byte-for-byte duplicates.
- All 29 parse as Python. Syntax validity does not establish that the snippets run correctly or independently.
- Many snippets require objects created by previous cells, including `train`, `X_tr`, `y_tr`, `params`, `skf`, `oof_raw` and `cat_cols`. They also reuse and mutate these names. They do not form a reproducible standalone workflow.
- The files contain print statements that would produce results, but no saved execution outputs or measured comparisons against V9. Imports of RandomForest, ExtraTrees, Ridge and other estimators do not establish that those models were actually evaluated.
- No supplied file implements a complete, verified V9 comparison, final submission export and matching executable delivery notebook.

The accompanying [source and data manifest](../improvement_v10_review/source_and_data_audit.json) records all source paths, full SHA256 hashes, line counts, imports, duplicates, official input hashes and the preserved V9 CSV/notebook hashes.

## Findings that affect model selection

1. **The claimed out-of-fold target encoding is not cross-fitted for training rows.** In [1793039118315, lines 24 onward](~/Downloads/gemini-code-1793039118315.py:24), category means are computed from all outer-training targets and applied back to those same training rows. [1793040782819](~/Downloads/gemini-code-1793040782819.py:30) repeats the pattern. A restaurant contributes its own target to its encoded predictors. Outer-validation targets are excluded from those means, so this is not a claim that the outer labels are directly leaked. It creates training self-label contamination and a training/inference mismatch. If revisited, use inner cross-fitting for training rows, then training-only maps for validation/test. Scikit-learn describes the distinction in its [cross-fitting example](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html).

2. **Several HGB versions treat category codes as ordered numbers.** [1793039347510](~/Downloads/gemini-code-1793039347510.py:1) replaces nominal categories with integer codes and constructs HGB without declaring categorical features. [1793041016975](~/Downloads/gemini-code-1793041016975.py:20) also codes the opening-date string as an arbitrary category number. Later [1793040782819](~/Downloads/gemini-code-1793040782819.py:38) does explicitly declare categorical columns, which addresses this part of the problem. Native HGB also has a per-feature category limit determined by `max_bins`; exceeding that limit must be checked, not assumed. See the [HGB API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

3. **Feature-only group statistics are computed before validation splitting.** [1793040565406](~/Downloads/gemini-code-1793040565406.py:52) concatenates all train/test rows, computes city counts, group means and frequencies, and only later splits for model validation. [1793039902514](~/Downloads/gemini-code-1793039902514.py:3) does the same for many group deviations and z-scores. These are target-free transductive statistics, not direct target leakage, but outer-validation feature values help build the representation. Fit maps on each training subset and freeze them for prediction. Check prediction-batch invariance as well as row permutation: permuting a complete batch alone will not detect dependence on the other rows in that batch.

4. **Low-turnover-only RMSE cannot select the submission.** Several files emphasize rows at or below 65M or 80M, including [1793039828263](~/Downloads/gemini-code-1793039828263.py:35) and [1793041023703](~/Downloads/gemini-code-1793041023703.py:9). These are useful diagnostics. They do not establish what targets occur in the hidden test set. Select using ordinary RMSE on every original validation target. Clipping training targets can be an experiment, but is not a repair justified by assuming large labels are incorrect.

5. **Retransformation corrections are fitted to training predictions.** The log variance correction in [1793039118315](~/Downloads/gemini-code-1793039118315.py:38) and mean-ratio correction in [1793040782819](~/Downloads/gemini-code-1793040782819.py:45) use fitted training predictions. They do not establish an appropriate correction for unseen restaurants. The variance formula also requires distribution assumptions that the files do not verify. If a correction is tested later, estimate it from inner out-of-fold predictions and assess the unchanged outer targets. This round will use the successful raw target and no correction.

6. **There are smaller reproducibility errors.** `log_weight` is accepted but ignored in [1793040782819](~/Downloads/gemini-code-1793040782819.py:18); its blend is hard-coded at 45/25/30. That file squares square-root predictions without first truncating negative transformed values at zero. The city function removes punctuation before checking sentinel `-1`, and includes an unreachable leading-space mapping after stripping. The larger pipelines assign arbitrary reference years 2015 or 2016 and sometimes fill unknown tiers with 2 before interactions. None is a measured improvement over the frozen V9 procedure.

7. **Dataset-origin assertions remain unverified.** Comments propose that these are transformed AMEO salary records and relabel certain variables as personality/aptitude scores. No source dataset, documented transformation, verified row linkage or execution evidence is supplied. The official data has 3,493 training plus 500 test rows; a row count or rounded target pattern alone does not identify its origin. Do not use those comments to rename variables, import presumed hidden labels or explain the leader’s score.

The rolling target statistic in 1793041016975 is explicitly dropped before fitting. It is an EDA diagnostic in that snippet; this review does **not** claim that it was passed into the model.

## What the official-data checks actually establish

| Check | Train | Test | Meaning |
|---|---:|---:|---|
| Raw city labels | 297 | 97 | Whitespace/case and spelling variants are present. |
| V7-clean city labels | 196 | 72 | Much city normalization is already implemented. |
| Labels after the conservative proposed literal alias map | 193 | 68 | A limited additional representation change. |
| Rows whose city token changes relative to V7 | 25 | 8 | Many changes are aliases; not every change expands coverage. |
| City rows unseen in training, V7 → proposed map | 0 → 0 | 24 → 20 | Four test rows gain known-city coverage; this does not prove an RMSE gain. |
| Zomato rating equal to zero | 13 | 0 | Outside the dictionary’s 1–5 range; test a missing-value interpretation with an explicit flag. |
| Invalid parsed opening dates | 0 | 0 | Date parsing itself is already correct. |

The map audit preserves the V7 aliases and adds selected literal aliases from 1793040565406. It does not apply that script’s blanket punctuation deletion. Ambiguous tokens and broad metropolitan pooling, such as mapping `navi mumbai` or `thane` to `mumbai`, are deferred. The exact proposed map is in the manifest; its effects above are feature-coverage measurements, not fitted-model results.

All 3,493 training targets are multiples of 100,000; 3,492 are multiples of 500,000. That does not justify rounding predictions. A conditional mean that minimizes squared error need not lie on the observed target grid.

There are 48 targets above 80M and a maximum target of 400M. The supplied scripts do not prove that these labels are corrupt. V9’s existing random confirmation gains were concentrated mostly in ordinary-turnover rows; the >80M group worsened on those assignments. Keep the full metric and segment diagnostics together.

## Useful ideas, ranked against work already completed

| Idea | Earlier work | Recommendation |
|---|---|---|
| A small, scale-aware quality/facility/event feature block supplied to TabPFN | V5 rich features and V8 structured Ridge contain some social/date interactions. V9 TabPFN used the 40-column V7-clean representation. | Highest priority: test the representation on the successful fixed TabPFN model, with a precise block ablation. Do not call every engineered feature new. |
| Additional conservative city view and explicit zero-Zomato treatment | V5/V7 already normalize text and ten city aliases. | Small, separately measured cleanup block. Preserve the original city view and do not assume zero means missing with certainty. |
| Training-subset frequency maps and target-free group deviations | Large transductive versions appear in the attachments; some simple counts appeared in V8. | One restricted extra representation, fitted inside each fold. |
| Raw/log/square-root/capped HGB ensembles and more model families | These families/target treatments already appear across V2–V8. V8 HGB/Ridge did not qualify; V9 EBM did not qualify. | Defer another broad search. |
| AMEO row matching, ID patterns, predicted test caps, target rounding | Speculation or diagnostics without a reproducible better model. | No main training budget or submission rule based on these claims. |

## File-by-file coverage

Each suffix below identifies `~/Downloads/gemini-code-<suffix>.py`. Every file was inspected; duplicated contents were also hash-verified.

| Suffix | Purpose and disposition |
|---|---|
| 1793038917092 | Data loading, target granularity and proposed AMEO origin. EDA; origin unverified. |
| 1793038921109 | Schema/range/missingness audit and endorsement-column spelling alignment. Useful, already covered. |
| 1793038925753 | Environment checks and speculation about hidden-test target limits. Keep environment checks; reject the limit assumption. |
| 1793038928480 | City/cuisine EDA and target summaries by cuisine. Diagnostic only. |
| 1793038947785 | Combined city counts and row/ID/feature-sorted target autocorrelation. No verified predictive-ID rule. |
| 1793038991194 | City vocabulary difference. Already covered. |
| 1793038993172 | Print prior city vocabulary; requires prior globals. |
| 1793039009620 | City/date/social/event/vibe/facility features and combined-data group deviations. Adapt selected ideas with fold-local maps. |
| 1793039118315 | Stratified HGB raw/log/capped comparison. Own-label target encoding; fitted log correction; filtered diagnostics. |
| 1793039347510 | Numeric-code HGB version. Same encoding/correction issues, plus nominal codes treated as ordered. |
| 1793039828263 | Pure-log comparison and low-turnover metrics. Repeated target encoding; no V9 evidence. |
| 1793039887567 | Full and filtered metrics for existing OOF vectors. No new model fit. |
| 1793039897406 | Exact duplicate of 1793039887567. |
| 1793039902514 | Extensive advanced features, category interactions and group z-scores. Too broad; subset ideas only, with fold-local maps. |
| 1793040357375 | Imports and input loading; not an evaluated LightGBM/XGBoost experiment. |
| 1793040358474 | Categorical vocabulary and numeric range/missingness audit. Useful diagnostic. |
| 1793040361521 | Category-count EDA. Already covered. |
| 1793040449983 | Cuisine target summaries and token counts. Diagnostic only. |
| 1793040501263 | ID ranges and row-block target statistics. Does not establish an inference-time ID feature. |
| 1793040507412 | Positions of missing values for selected columns. Diagnostic only. |
| 1793040523135 | Exact duplicate of 1793040507412. |
| 1793040565406 | Most useful source of scale-aware features, facility counts and city ideas. Correct its global maps and separate semantic assumptions. |
| 1793040782819 | Engineered HGB raw/log/sqrt ensemble. Own-label encoding, fitted correction, ignored parameter and fixed arbitrary blend. |
| 1793040786727 | Target grid checks. Do not turn this into prediction rounding. |
| 1793041010817 | Raw categorical EDA. Already covered. |
| 1793041013037 | City and train/test numeric range/missingness checks. Useful diagnostic. |
| 1793041016975 | Ten-fold HGB/target-variability diagnostic. Drops rolling target statistic correctly; uses arbitrary ordinal category/date codes. |
| 1793041021982 | Column listing and AMEO speculation. No verification of origin. |
| 1793041023703 | MAE/median error and filtered RMSE diagnostics. Keep full RMSE primary. |

The resulting [proposed V10 plan](../improvement_v10_review/V10_Plan.md) defines four representations and a fixed validation/acceptance policy. It has not been launched. No evidence in these files supports promising a first-place score.
