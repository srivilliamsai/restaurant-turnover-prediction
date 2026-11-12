# V20 proposal: test the two families that never reached accuracy evaluation

Prepared October 29, 2026, IST. **Execution complete: all three engineering pilots passed, 30 development folds and 20 blends finished, and no candidate qualified. Retain V14.** See execution_status.json and transport_run/status.json for the current state. The protocol below is preserved.

**Recommendation: recover TabICLv2 first, then test Kumo Small in its own clean environment.** V14 remains the best observed portal submission, **RMSE 11,736,110.7877709**. V20 is a bounded experiment with a plausible technical reason to try it; there is no evidence yet that it improves V14 or reaches a particular leaderboard rank.

## Why V19 did not qualify

The [independent audit](evidence_audit.json) reconstructed 20 V19 candidate fold outputs, ten matched P35 control outputs and all 15 blend comparisons. Input, fold-membership, training-target and prediction hashes were checked. Reconstructed scores match the saved scores exactly in the audit environment. The exact scored V14 CSV and notebook hashes also match their score record.

| Evidence | Finding | Implication for V20 |
|---|---|---|
| Best V19 blend | 75% V14 + 25% Causilo; local gain **0.013192%**, approximately INR 2,592 on average | Far below the existing 0.20% floor, approximately INR 39,299 at this local baseline. Do not spend another round on neighboring Causilo weights. |
| Causilo / TabDPT | Both completed accuracy testing; neither qualified. Residual correlations with V14 are approximately **0.996 / 0.989**. | They share much of V14's error. Correlation is a diagnostic, not an independent selection rule. |
| Kumo Small / Medium | Both stopped at `numba.cuda.types.NPDatetime`; **accuracy was never tested**. | This is an environment failure, not evidence that Kumo predicts poorly. |
| TabICLv2 in V11 | Stopped at a row-permutation prediction difference of **INR 3.814697265625**, before the remaining engineering checks and accuracy testing. | That particular difference is below V19's already adopted absolute INR 10 numerical tolerance. Recheck the full lifecycle under the current rule. It is not already a passed model. |
| Error concentration | 48 of 3,493 labels exceed INR 80M and account for approximately **68.1%** of V14 squared error; the five largest errors account for **34.8%**. | Keep original-scale RMSE and report concentration. The extreme labels have not been established as errors. |

V14's local development RMSE is about **19.649M**, while its observed portal RMSE is **11.736M**. These refer to different rows. Their difference does not prove label corruption, train/test shift, a portal bug or a useful clipping threshold. The audit found zero exact duplicate predictor rows in training and zero exact train/test predictor matches; it did not test every possible semantic or approximate duplicate. Raw marginal covariate diagnostics are saved, but cannot reveal hidden test targets.

V13 already tested training-label caps and a square-root target; V7 already tested turnover bands, tier specialists and nested residual models. Repeating those ideas without a materially different, predeclared hypothesis is low priority. Deleting high labels, using hidden test labels or choosing a predictor from a restaurant's true target band is outside this plan.

## Internet research and candidate order

1. **TabICLv2 is first.** Its [official implementation](https://github.com/soda-inria/tabicl) supplies a regression checkpoint, configurable numerical precision and attention, and fitted-state persistence. The original V11 code already constructed and ran it on this dataset. Reconsidering the cent-level rejection under the numerical standard established before V19 results is justified. No accuracy or full-stability conclusion follows from that observation.
2. **Kumo Small is second.** The [NVIDIA package](https://github.com/NVIDIA/structured-data-models) treats cuDF as optional. The exact source inspected in V19 includes an Arrow/CPU fallback for CUDA string comparisons when cuDF is absent. A clean environment without inherited optional GPU dataframe packages is therefore a specific repair hypothesis. It still needs a real end-to-end pilot; a compatible Numba version or successful import alone is insufficient. The [regression API](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.KumoTabular) returns quantiles, so freeze the point prediction explicitly.
3. **Defer the other leads.** [Google TabFM](https://github.com/google-research/tabfm) supports regression, but its documented default context is only **100 rows**. It must not silently replace the intended approximately 2,794 training rows with a small sampled context. [LimiX-2M](https://huggingface.co/stable-ai/LimiX-2M) is a genuine 2M-parameter regression model, distinct from 400M-parameter LimiX-2, but its published setup and preprocessing need further T4 and train-only-state verification. Parameter count does not establish feasibility. TabSTAR and TabSwift remain lower-priority integration work; their V19 source review is preserved.

These priorities are an inference from local failures and official interfaces, not a claimed external ranking. See the [research register](RESEARCH_SOURCES.md) for evidence and limitations.

## Three fixed settings

Keep V14's exact forty-column builder, city corrections and all original labels. Exclude registration number and turnover from predictors. Fit categorical vocabularies and every learned transformation on outer-training rows only. Use raw targets in INR millions, evaluate in original INR, and keep the baseline's fixed nonnegative output floor. Feature additions, label caps, alternative target losses and tail routing are not extra selectable arms in V20.

| ID | Recipe |
|---|---|
| **I8** | Original V11 `tabicl==2.2.0`, official `tabicl-regressor-v2-20260212.ckpt`, eight estimators, seed 10242, `norm_methods=None`, Latin feature shuffling, ensemble `batch_size=1`, `kv_cache=True`, CUDA, `use_amp=False`, `use_fa3=False`, two CPU threads. Preserve the V11 training-fitted categorical adapter and other resolved defaults. Use the predictive mean. |
| **KS8-CLEAN** | Exact NVIDIA source commit **2b1696f4fdc22004cc757c44703a69cf326c0a11**, Small regression checkpoint from the V19 checkpoint register, eight estimators, estimator batch size one, seed 10242, unchanged reviewed recipe. Construct typed tables on CPU through the public Arrow interface, then move through supported tensor/device APIs. Keep all context rows. After the recipe's inverse target transform and 20% trimmed estimator aggregation, average the 999 quantiles in float64. This is a fixed grid approximation to the mean. |
| **I16** | Same recipe as I8, with sixteen estimators. Optional admission decided from total measured workload before any development scores. Do not assume I8 outputs can be recovered from an I16 fit without proving identical transformations and members. |

Kumo Medium is deferred. This avoids multiplying a new dependency repair into another size search. Do not add another family after viewing development scores.

Use separate candidate virtual environments without system site packages; do not change the V14 environment. Prefer the already exercised Python 3.12 / PyTorch 2.11 CUDA stack where the pinned package constraints permit it. Record the actually resolved versions. For Kumo, establish that `cudf` and the failing optional Numba/CUDA chain are absent from the child environment; do not inherit them through `PYTHONPATH`. Install only the required model/data packages. Do not monkey-patch `NPDatetime`, hide an import exception with a dummy module or replace NVIDIA's preprocessing with an unreviewed numerical shortcut.

Reuse the known Kumo source and checkpoint identities. For TabICL, resolve the official regression file to an immutable Hub revision and byte hash before running it. Lock source/package hashes, dependency versions and downloaded weight bytes before development. The online main-branch documentation is evidence for planning, not an immutable execution lock. No new weights were downloaded in this planning task.

## Engineering admission before accuracy

1. Verify the four original input hashes and exact scored V14 artifacts. Reconstruct/reproduce control provenance and retain the original-Mac V7 boundary. A fresh Linux V7 fit is not interchangeable with the scored original.
2. Inspect the actual free GPU and available memory on the selected account. Pilot in order **I8, KS8-CLEAN, I16**. Each admitted setting must complete approximately 2,794 training / 699 query rows and 3,493 training / 500 query rows. Measure fit, predict, serialization, allocated/reserved GPU memory and host memory; keep practical GPU headroom.
3. Check finite outputs, target/ID exclusion, missing and unseen categories, stable ID/order mapping, reordered queries, different query batches, at least twelve individual queries, and a fixed query after changing other query rows. Test saved-state reload and a genuinely fresh fit. Every query uses the same training-fitted transformations and context. Padding, if needed, must use one fixed training-feature row and must be frozen before development.
4. Apply the numerical rule already used in V19: new-candidate prediction comparisons use **absolute tolerance INR 10 plus relative tolerance 1e-6**, with **pilot RMSE drift at most INR 10**. Preserve the historical V14 control requirement of less than INR 0.01. Do not round outputs or change thresholds to pass a failure. Material dependence on other query rows still excludes a candidate. The INR 3.81 V11 observation satisfies only one part of this rule; it does not excuse the other checks.
5. Save full raw failures. Allow at most **45 minutes of the shared engineering allowance** for the Kumo clean-environment repair. If the official path still fails, exclude it transparently. Do not spend the campaign searching arbitrary combinations of Numba, RAPIDS and CUDA releases.
6. Forecast the complete remaining workflow with **30% contingency**, including confirmation controls on Mac, candidate confirmation, full-data fits and fresh notebook execution. Admit I16 only if this fits. If more reduction is required, drop I16 first, then KS8-CLEAN; engineering failure of I8 does not preclude a passing Kumo-only campaign. Freeze the admitted settings before development accuracy is revealed.

Import success is not engineering admission. Engineering admission is not a prediction-quality result. No GPU, account or installed-environment check has been performed for V20 during planning.

## Fixed development search and diagnostics

Use five folds on the exact inherited assignments **42 and 71**, with verified matched V14 out-of-fold controls. All admitted settings must finish both assignments before selection. Selectable routes are each admitted setting individually and, only when both exist, the fixed **50% I8 + 50% KS8-CLEAN** average. I16 is not added to that average; it represents the same family as I8.

For each route M, evaluate exactly:

```text
B = V14 = 0.25 * original_V9 + 0.75 * original_P35_8
candidate = (1 - w) * B + w * M
w = 0.10, 0.25, 0.50, 0.75, 1.00
```

Maximum: **30 new development fold outputs and 20 route/weight comparisons**. The zero-weight baseline is the fallback. No continuous optimizer, learned router or additional stacking layer is selected from these same outputs.

Advance at most one candidate, requiring positive gain on both assignments and mean relative RMSE gain **at least 0.20%**. Rank qualifiers by mean relative gain, lower new-model share, fewer components, then route ID. Freeze the entire recipe and blend before confirmation. Do not lower the qualification floor to make a version number produce a CSV.

Report full-label RMSE, per-fold gains, and squared-error contributions in <=40M / 40–80M / >80M bands. Also report residual correlations on all rows and on <=80M rows, and sensitivity after removing V14's five largest-error rows from the diagnostic calculation only. These summaries explain concentration and stability; they neither delete training/validation labels nor select a new rule. A real overall improvement need not improve every tail subgroup.

## Confirmation and delivery

Confirm only the locked finalist on three additional five-fold random assignments **920201, 920202 and 920203**, plus inherited canonical-city and target-stratified assignment **27421** checks. These are new partitions of historically examined labels, not an untouched holdout.

Rebuild matched V14 controls for the new random assignments: **15 original-platform V7, 15 T9 and 15 P35_8 fits**. Reuse the city/stratified controls only when their exact row/fold/input/model hashes match. A selected one-family route needs **25 candidate confirmation folds**; the two-family average needs **50**. Budget all 45 random-control component fits and the slowest admitted finalist before development starts.

Keep the established confirmation rule: all three random gains positive, mean relative random gain >=0.20%, positive lower 95% paired-row gain bound from 5,000 resamples at seed 8319 using the same sampled restaurants across random assignments, and <=1% deterioration on each city/stratified check. The interval is conditional robustness evidence and does not erase repeated-search uncertainty. Do not retune or confirm a second candidate after failure.

Only a qualified finalist receives a fresh full-data fit, standalone executed notebook, 500-row CSV, exact environment locks, source/checkpoint hashes and measured report. Verify original test IDs/order, headers, finite nonnegative INR values and fresh notebook/CSV agreement under the frozen numerical rule. Cached expected values are verification witnesses, not a substitute for inference. A V14-derived final artifact must explicitly document its original-platform legacy component and its independent verification.

At most one qualified **manual** portal trial; associate its actual score with the exact CSV hash. A local pass alone does not replace V14. Retain the best observed portal score and its matching notebook. Keep the one-time coding upload dependent on the actual competition state.

## Proposed resource cap and stop rule

**Six elapsed hours total**, including implementation, setup, downloads, failures, interruptions, recovery, confirmation and delivery. Reserve **1.5 hours for verification and delivery**. Engineering has a one-hour ceiling inside the remaining 4.5-hour shared research pool; the 45-minute Kumo repair ceiling is part of that hour, not extra time. Shrink admission before scoring if the measured complete forecast cannot fit. Stop early once the declared outcomes are known.

This is a proposed execution cap; no V20 execution ledger has started. Check the current portal cutoff and timezone before execution and shorten the campaign to leave at least one hour before it. Older notes contain differing deadline assumptions and do not establish the current cutoff. Free GPU allocation and uninterrupted duration are not guaranteed; checkpoint complete folds and respect the [current Colab limits](https://research.google.com/colaboratory/faq.html).

**If these untested families also fail, retain V14 and close this model-search campaign.** A later round would need a distinct, evidenced data/feature hypothesis rather than another neighboring ensemble weight or an ever-growing model list. V20's honest outcomes are either a reproducible qualified finalist or a documented stop with the exact V14 submission preserved.
