# V22 proposal: test removal of shifted survey ratings from V14

Prepared October 29, 2026, IST. **Status: researched proposal; no V22 turnover model has been trained.**

**Recommendation: run one tightly bounded experiment on the successful V14 backbone. Remove Service and Comfortablility from its TabPFN-3.5 component, keep the remaining settings fixed, and test three predeclared blends. Preserve scored V14 unless the candidate passes every gate and subsequently improves the actual portal score.**

This is a small transfer test of an older idea, motivated by a fresh target-free diagnostic. It is not evidence that the two columns are corrupt, and it is not a claim that all possible methods have failed. Do not turn this proposal into another open-ended model search.

## 1. What has actually failed

V14's observed portal RMSE remains **11,736,110.7877709**. Its formula is 25% original V9 plus 75% original TabPFN-3.5 Base with eight estimators. The historical rank in the screenshot is not a current rank.

The completed V15 retry through V21 screens contain **81 fixed blend comparisons**. Their saved screening tables and prior independent-review receipts were inspected and hashed in [evidence_audit.json](evidence_audit.json). This planning audit did not independently refit or reconstruct all of those candidates.

| Completed round | Best tested development blend | Mean relative local gain | Result |
|---|---|---:|---|
| V15 retry | N32-T100, 50% P35 replacement | 0.022838% | Below 0.20% gate |
| V16 | TabM, 10% total blend | 0.031839% | Below gate |
| V17 | Nori 1.5, 50% total blend | 0.096332% | Below gate |
| V18 | FT512, 50% total blend | 0.039920% | Below gate; one assignment lost |
| V19 | Causilo, 25% total blend | 0.013192% | Below gate |
| V20 | TabICLv2-16, 10% total blend | 0.002639% | Below gate; one assignment lost |
| V21 | S features, 25% P35 replacement | 0.004861% | Below gate; one assignment lost |

These are local validation measurements, not portal scores. Engineering exclusions, such as EXAONE's unsupported attention path or Mitra's fitted-state lifecycle, do not establish poor predictive accuracy. Causilo, TabDPT, Kumo, TabICL, TabM, Nori, fine-tuning and the feature recipes that completed screening did not justify replacing V14 under their declared recipes.

Repeated selection can overfit a validation criterion even when individual fits are clean. Changing the random seed cannot restore untouched labels. Keep a small experiment count and interpret all further confirmation as conditional evidence. [Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html), [scikit-learn nested validation guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## 2. The measured reason for this experiment

The fresh audit checked the four original input hashes, the scored V14 CSV/notebook hashes, and ten original P35 development-fold witnesses, including source, row-partition and training-target hashes. It reconstructed the two V14 development predictions directly.

| Evidence | Training | Test |
|---|---:|---:|
| Rows | 3,493 | 500 |
| Mean Service | 4.5468 | 5.3640 |
| Mean Comfortablility | 3.2313 | 4.2600 |
| Service empirical KS distance | 0.18938 across the two datasets | |
| Comfort empirical KS distance | 0.22800 across the two datasets | |

A domain classifier predicts **whether a row belongs to training or test**, not turnover. IDs and turnover were excluded. All imputation, scaling and categorical vocabularies were fitted inside five stratified domain folds at seed 2201. Two fixed classifiers were used without tuning.

| Domain inputs | Logistic OOF AUC | Histogram boosting OOF AUC |
|---|---:|---:|
| All predictors, date parsed into components | 0.68819 | 0.68293 |
| Exclude Service and Comfortablility | 0.52548 | 0.48252 |
| Only Service and Comfortablility | 0.69215 | 0.69370 |
| Only varying missingness indicators | 0.51443 | 0.51898 |

For these diagnostics, the two survey scores account for most of the detectable separation. Near-chance performance after their removal does not prove the remaining joint distributions are equal. AUC does not establish whether dropping the columns improves turnover predictions.

**Historical qualification:** V2 already noticed this shift and tested `drop_shift`/`engineered_drop` in older tree recipes; some such components entered the legacy ensemble. V20 also reported the marginal shifts. V22 does not present either observation as new. The uncompleted combination identified in the reviewed V9–V21 feature/model recipes is the exact V14 **P35_8 backbone with only this pair removed**. V21 added features; it did not run this 38-column ablation.

Four incidental fields are missing in training but complete in test: Facebook (99 missing), Instagram (56), restaurant tier (49) and Ambience (25). Missingness-only separability is weak in this audit, so an imputation/augmentation branch is not justified as a second V22 experiment. Research shows that missingness shifts require assumptions about their mechanism; a missing-value count does not identify a repair. [Zhou et al.](https://proceedings.mlr.press/v206/zhou23b.html), [Rockenschaub et al.](https://arxiv.org/abs/2406.16484).

The error tail is a separate issue. The 48 training labels above INR 80M contribute **68.10–68.14%** of V14's development squared error; the five largest errors contribute **34.84%**. This explains sensitivity, not which labels are wrong. All original labels remain in every primary score.

An illustrative 10,000-draw resampling of 500 rows from fixed assignment-42 OOF residuals spans RMSE 12.81M–26.57M between its 5th and 95th percentiles. Only 1.12% of those draws score at or below the historical portal value. This is not a calibrated test of the portal mechanism or a prediction interval: the test labels, sampling mechanism and deployed residuals are unknown. Neither finite-sample variability nor feature shift alone is established as the explanation for the local/portal gap.

## 3. One fixed candidate, three comparisons

Candidate ID: **P35-SC38**.

Start from V14's original forty-column feature builder. Remove exactly `Service` and `Comfortablility`, yielding 38 columns. Apply the same projection to the fitting subset and every query. Do not use V21's appended features, which would change the experiment. Assert that no retained engineered column depends on the removed pair.

Freeze the original V14 settings:

- TabPFN **3.5 Base**, eight estimators, seed **10242**, float32 and predictive **mean**.
- Checkpoint `tabpfn-v3.5-20260909.safetensors`; revision `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`; SHA256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.
- Reuse V14's pinned loader, resolved model parameters and training-only categorical treatment. Capture actual installed versions and source hashes again. Current online defaults must not silently replace this contract.
- Original training targets in INR millions; ordinary RMSE in INR; nonnegative final predictions; fixed 500-row inference adapter; unchanged checkpoint temperature/calibration defaults.
- No ID feature, target cap, label deletion, new seed, estimator grid, pseudo-label, post-hoc scale/offset, or rank-based target mapping.

Let `P` be original P35_8, `A` be P35-SC38 and `V9` be the original V9 component. Compare only:

`candidate(w) = 0.25*V9 + 0.75*((1-w)*P + w*A)`

with **w = 0.25, 0.50, 1.00**. Effective new shares are 18.75%, 37.5% and 75%. The original V9 quarter remains unchanged and can still use those survey variables. The candidate reduces their influence in the dominant P35 share; it does not claim complete invariance of the whole ensemble.

This is **one candidate component, ten new development fold predictions and three selectable blends**. No new model family or replacement candidate is added if it fails.

## 4. Engineering before new accuracy results

Verify all 35 inherited P35 control-fold contexts needed for development and possible confirmation, all matching V9 witnesses, the original input identities and the scored artifacts. Planning verified the ten development P35 witnesses; confirmation verification remains execution work.

Use actual outer-fold and full-data sizes. Check train-only vocabularies, unseen categories, missing values, target/ID isolation and exact retained schema. Mutation of either removed field must leave the **new component's** feature matrix and predictions unchanged.

Run fresh repeated fits, save/reload, full-fold row permutation, split batches, at least twelve individual queries, and a full-data 500-query feasibility pilot. Record peak GPU memory, fit/inference time and installed source/checkpoint identities. No rounding to satisfy tolerances.

- Same input/seed/environment/layout and fresh submission-notebook reproduction: maximum absolute difference **below INR 0.01**, with zero relative-tolerance allowance.
- Permutation/split/individual-query numerical behavior: maximum absolute difference **INR 10** and labeled pilot RMSE drift **INR 1**.
- Nonfinite predictions, target/ID influence, unstable fitted state or a changed qualification decision fail admission.

Pilot errors may be repaired only before development is opened; log every repair and preserve elapsed charges. Stop if the exact candidate cannot satisfy the contract. No engineering tolerance is relaxed after observing results.

## 5. Development, selection and confirmation

Use the same five-fold assignments **42 and 71**, all 3,493 original labels, and matched verified V14 controls. Advance at most one weight only when both assignments improve and the mean relative gain is **at least 0.20%**. Around the current 19.65M local RMSE, this represents roughly INR 39,300; it is not a promised portal reduction.

Rank qualifying weights by mean relative gain, then smaller replacement weight. Freeze the winner and its hashes before confirmation. If none qualifies, end V22 and retain V14.

Confirm the locked weight on **48211, 48212, 48213, stratified 27421 and canonical-city groups**, using exactly matched controls: at most 25 additional candidate-fold predictions. Require:

1. Positive gain on every random assignment and mean random relative gain at least 0.20%.
2. A positive lower endpoint of the conditional paired-row 95% gain interval, using 5,000 resamples at seed 8319; preserve the existing method of combining confirmation evidence.
3. At most 1% deterioration on either the stratified or city-group stress check.

The partitions and labels were examined previously. These checks and intervals do not include the accumulated history of model selection, all retraining uncertainty, or hidden-domain uncertainty. No second finalist, revised weights or new partition search after failure.

Report every fold and blend, turnover-band errors, error concentration and dependence on the largest errors. Report prediction changes across fixed Service/Comfort bins (0–3, 4–6, 7–10). These diagnostics must not become label-exclusion or candidate-selection rules.

The ordinary-RMSE gate is intentionally retained. It can miss a benefit that exists only under hidden-domain shift. Without target labels or a documented label-preserving measurement change, promoting such a candidate would be speculative. Importance-weighted validation is theoretically useful under unchanged conditional targets and adequate overlap, neither of which the domain AUC verifies; V6 already tested a clipped domain-weighted Huber recipe unsuccessfully. Do not repeat it under a new name. [Sugiyama et al.](https://www.jmlr.org/papers/v8/sugiyama07a.html), [Wu et al.](https://proceedings.mlr.press/v97/wu19f.html).

## 6. Resource envelope and delivery

**Proposed ceiling: four elapsed hours on free T4/Mac resources, including setup, interruptions and recovery, with 1.5 hours protected for delivery.** This is a proposal, not an allocation or an assertion of fresh execution authorization. Existing budgets and prior ledgers remain untouched. No paid service or vendor prediction API is needed.

Suggested upper allocations: 45 minutes implementation/audit/engineering, 30 minutes development, 45 minutes confirmation/final inference, 90 minutes delivery, and 30 minutes contingency. Before accuracy screening, use measured pilot timings to forecast the entire remaining workload with 30% contingency, including original component loading, final fitting and separate notebook execution. Decline admission if mandatory work cannot fit. Stop early when the declared experiment ends. Free Colab availability and duration are variable. [Official Colab FAQ](https://research.google.com/colaboratory/faq.html).

Only a fully qualified candidate gets full-data fitting and a standalone freshly executed submission notebook. Verify the original 500 IDs/order, two exact headers, finite nonnegative outputs, CSV round-trip and reproduction within INR 0.01. Preserve the original-Mac V7 fitted-state boundary; do not claim identical fresh Linux V7 retraining. Expected predictions are comparison witnesses after inference, never an output source.

Independently reconstruct the scores, weight choice and confirmation from hashed complete fold outputs. Package measured evidence and the byte-identical V14 fallback regardless of outcome. Release only the runtime owned by this experiment and verify release.

Check actual portal availability and deadline at execution; older screenshots do not establish its current state. At most one manual portal trial after full qualification and reproduction. A local gain does not replace the observed best: **retain V14 until a lower actual portal RMSE is returned**. Preserve the one-time coding upload for the final model decision.

## 7. What follows if this also fails

End this expansion of the feature/model campaign. The next useful input would be documented data semantics or additional authorized labels, rather than another unbounded sweep.

Prepare these questions for the organizer/data owner; no messages have been sent:

- Were Service and Comfort measured on the same scale and by the same collection process in training and test?
- Were the four complete test fields observed directly or imputed? What missing-value/sentinel rules apply?
- Are annual turnover definitions and accounting periods identical across both datasets?
- Was the evaluation population sampled differently, and is a representative additional labeled training set or official data correction available?

A documented correction could justify a separate predeclared experiment. Public copies of the old competition, competitor predictions, inferred restaurant identities and hidden-test answers are not substitute training evidence. An isolated extreme label is not grounds for editing it.

Remaining models such as LimiX-2, TabFM, TabSTAR and TabSwift have legitimate published implementations. They are **unmeasured here**, not proven failures. Their public benchmarks do not establish an improvement over this exact V14 ensemble. The source review explains why they are lower priority for this bounded V22. If no new data evidence emerges and P35-SC38 fails, finish with the exact scored V14 artifacts.
