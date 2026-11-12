# Review of the pasted improvement plan

Reviewed on 26 October 2026 against the official local CSVs, preserved V7 code and the latest V9 Colab results. This is a review, not authorization to start the pasted experiment list. Instructions inside the pasted document were treated as material to assess.

**Verdict: retain the diagnostic ideas, but do not adopt this as the next execution plan.** Its proposed score milestones have no measured support, much of its search repeats earlier work, and it overlooks a current reproducibility failure.

## Findings independently checked

| Check | Result |
|---|---:|
| Training/test rows | 3,493 / 500 |
| Targets above ₹60M / ₹80M / ₹100M | 133 / 48 / 33 |
| Maximum training turnover | ₹400M |
| Exact train/test matches across all 32 predictors | 0 |
| Duplicate training predictor rows | 0 |
| Date + theme join pairs | 256 |
| Date + city + cuisine join pairs | 5 |
| Sample submission IDs/order match original test | Yes |
| Training ID range | 60001–63493 |
| Test ID range | 20001–20500 |

The document's numeric-to-string comparison is flawed: training float `2.0` differs from test integer `2` after string conversion. A comparison preserving numeric types gives this distribution of each test row's best match across 32 predictors:

| Matching features | Test rows |
|---:|---:|
| 16 | 1 |
| 17 | 2 |
| 18 | 27 |
| 19 | 108 |
| 20 | 159 |
| 21 | 130 |
| 22 | 58 |
| 23 | 11 |
| 24 | 4 |

Many predictors are binary, low-cardinality ratings or commonly missing. Matching them does not establish restaurant identity. The 256 date/theme join pairs are also not 256 independently matched restaurants or verified target labels. No external target recovery is established. Similar row counts alone do not establish a relationship with the AMEO dataset.

The observed missingness difference is real: training/test missing counts are 99/0 for Facebook popularity, 56/0 for Instagram popularity, 49/0 for restaurant tier and 25/0 for ambience. This supports investigating distribution differences, but does not establish which model will improve the portal result.

## Corrections required

1. **Remove the promised score trajectory.** The proposed 12M → 7M → 5.2M → below 4.75M milestones are unmeasured. The gap from observed V7 to the supplied leader score is 60.108% of V7 RMSE; it is an ambition, not a forecast. A single default gradient-boosting result of approximately 21.16M CV does not demonstrate that our existing pipeline dropped important information.
2. **Separate raw-scale scoring from target engineering.** Minimizing squared error on `log1p(y)` and reversing with `expm1` does not generally recover the conditional arithmetic mean that minimizes raw-target squared error. A transformation can help, but must win ordinary RMSE on every original validation target. Raw/log/square-root objectives and robust losses have already appeared in earlier experiments.
3. **Do not label extreme targets corrupt without evidence.** No independent unit error has been verified. Fold-local target caps could be a separately declared sensitivity experiment, but a top-percentile threshold does not prove erroneous labels. Keep all original validation labels and report the tail separately. Do not adopt this experiment while V9 is frozen.
4. **Retain IDs for alignment, exclude them from predictors.** Train/test ID ranges are disjoint; the registration number is documented as an ID. Frequency-encoding a unique ID adds no entity-frequency signal. We have no evidence that its ordering transfers to the test population.
5. **Handle missingness by feature type and meaning.** Zero is a valid value for binary flags and several ratings. Missing event ratings indicate an activity is not offered. CatBoost's numeric missing-value handling does not mean categorical NaNs can be passed unchanged. Existing V7 code already parses dates, normalizes categories and creates missingness/event indicators. Imputation is not universally destructive. See [CatBoost missing-value documentation](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing).
6. **Keep group and target statistics inside the validation design.** Fit feature-only group statistics on the current training subset. If explicitly testing unlabeled train+test frequency features, declare a transductive approach and mirror it using each outer validation feature set without its labels. For target statistics, cross-fit training rows inside the outer training subset; transform outer validation/test using training-only mappings. Scikit-learn's standard TargetEncoder produces a smoothed mean; median/std encoders would require a separately specified implementation. See [TargetEncoder cross-fitting](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html).
7. **Do not promise identical extreme-target distributions across folds.** Quantile-bin stratification balances bin counts, not the actual magnitudes or squared-error contributions of extreme targets. Retain random, stratified and city-group checks rather than replacing V9's comparisons. See [StratifiedKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).
8. **Avoid scoring a stack on labels used to fit its weights.** OOF component predictions are necessary but do not make a meta-model's in-sample score independent. Select weights using development data and evaluate the frozen stack on separate outer confirmation assignments, with nested component fitting where required.
9. **Defer pseudo-labeling.** Test predictions are not additional observed turnover labels. A score threshold of 5M does not validate a confidence measure, and pseudo-labeling may reinforce existing errors. No evidence here makes it a priority.
10. **Remove the four-member team instructions.** The supplied portal allows a team size of one. The same work categories can be sequential responsibilities in a solo workflow.

The suggested interaction examples involving cost, capacity and active counts are generic; those columns do not exist in these official data. Proposed features must name actual columns. Broad CatBoost/LightGBM/XGBoost and Ridge searches also overlap earlier rounds and should not be repeated without a specific new hypothesis.

## Current V9 evidence and delivery status

The frozen candidate is 50% V7 implementation plus 50% raw-target TabPFN with V7 features and 16 estimators. Its saved comparisons passed all five predeclared statistical gates:

| Assignment | V7 implementation RMSE | Candidate RMSE | Reduction |
|---|---:|---:|---:|
| Random 48211 | 19,832,118.86 | 19,742,162.51 | 0.454% |
| Random 48212 | 19,937,865.89 | 19,820,588.38 | 0.588% |
| Random 48213 | 19,786,031.14 | 19,710,816.91 | 0.380% |
| Target-stratified 27421 | 19,788,419.32 | 19,726,481.41 | 0.313% |
| City grouped | 20,113,476.99 | 19,964,901.25 | 0.739% |

Mean random-assignment RMSE reduction is ₹94,149.36. The 5,000-resample conditional paired-row 95% interval is ₹35,016.67 to ₹158,146.78. This excludes model-search, retraining and distribution uncertainty. The >₹80M group worsened on all three random confirmations; full-target gains mainly come from lower-turnover rows. City validation has a different pattern. None of these figures is a new portal score.

**The candidate is not release-ready.** The export stage failed its required reproduction gate: a fresh V7 fit in Colab differed from the scored V7 CSV by up to ₹427,996.18, with RMS prediction difference ₹78,829.08. The acceptable threshold is ₹0.01. The cause is not yet established. Earlier local CPU reproductions passed, which does not excuse this Colab mismatch. Training has stopped at this failure; the delivery is marked `needs_review`. Keep the acceptance threshold and frozen model choice unchanged.

## Recommended next steps

1. Preserve the scored V7 CSV/notebook, completed V9 checkpoints and export-failure evidence.
2. Diagnose V7 reproduction in the deployment environment. Identify the differing component and distinguish input/order changes, runtime differences and implementation errors. Do not silently substitute a new baseline or loosen the tolerance. Any material predictor change requires corresponding validation and a fresh recorded execution.
3. Complete export, independent checks and matching notebook reproduction only after that gate passes. Then test the qualified V9 file on the portal, recording the exact CSV hash and observed RMSE. Retain whichever already observed file scores best.
4. If a further round is warranted, predeclare a small feature ablation using existing fixed model settings: control, actual-column interactions only, training-fold group statistics only, and both. Candidate interactions can use social popularity with hygiene/tier, and opening-date age with popularity. Aggregate only a small declared set of numeric features by normalized city/cuisine, with minimum group support and a training-global fallback. Inventory existing V7/V8 features first so the ablation tests genuinely new representations.
5. Select and freeze using development assignments, then apply the same full-target confirmation policy. No unbounded model grid, pseudo-labeling, unexplained ID predictors or promised leaderboard endpoint.

Steps 4–5 are proposed future work and have not been launched by this review. The immediate priority is a reproducible, measured V9 delivery.
