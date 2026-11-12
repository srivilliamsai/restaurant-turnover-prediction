# Proposed V14: complete two previously blocked backbone tests

Prepared October 28, 2026, IST. Status: research and planning only. This document proposes an eight-hour execution cap; it does not start a Colab session or claim a new submission.

**Recommendation: test actual TabICLv2 and TabPFN-3.5 Base, after resolving their V11 engineering exclusions. Preserve scored V9: portal RMSE 11,904,515.9864996.** More target caps, square-root variants and extra estimators have weak measured support here.

## What the evidence says

| Round | Measured outcome | Implication for V14 |
|---|---|---|
| V9 | Improved the observed portal RMSE to 11,904,515.99 | Keep its CSV and matching notebook as the control and fallback. |
| V10 | Portal RMSE 11,925,736.50, worse by 21,220.51 despite positive local gains | Small local improvements do not justify an expected portal improvement. |
| V11 | Actual TabPFN-2.5, best development mean relative gain 0.03322%; TabICL excluded; 3.5 load failed | Neither TabICL nor actual 3.5 completed accuracy screening. |
| V12 | More TabPFN estimators/seeds: best 0.01179%; RealMLP did not qualify | Further estimator scaling has low priority. |
| V13 | Q99/Q995/SQ2: no blend improved both development assignments | Keep original targets; end this target-treatment search. |

The V13 screen was independently recomputed for this plan from all 30 complete candidate fold outputs and all 12 blends. The closest SQ2 blend, at 25%, gained INR 5,193.64 on seed 42 and lost INR 3,464.43 on seed 71. Its mean relative gain was only 0.00438%, versus the fixed 0.20% requirement. Confirmation was correctly skipped. See [review evidence](review_evidence.json) and its [reproduction script](review_v13.py).

The local V9 random-fold RMSE is approximately 19.7M, while its observed portal RMSE is 11.9M. In the five reviewed random assignments, the 48 labels above 80M contribute 68.0–68.3% of squared error; the five largest errors contribute about 35%. These measurements explain score sensitivity. They do not prove label errors, a hidden-test sampling mechanism or an achievable leaderboard score. Keep every original label in the primary metric.

## Why these two tests remain useful

**TabICLv2:** V11 stopped it at a row-permutation difference of INR 3.814697265625 under an INR 0.01 check. That is an engineering exclusion, not an accuracy failure. PyTorch documents that different batching and operation orders can produce floating-point differences. Determine whether the difference is numerical or substantive before excluding the family. [Official TabICL regression API](https://tabicl.readthedocs.io/en/latest/api.html), [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

**TabPFN-3.5 Base:** V11 recorded an UnpicklingError and substituted 2.5. Its 0.03322% result therefore says nothing about 3.5 accuracy. The official 3.5 model card supplies one synthetic-pretrained checkpoint for regression and classification. It remains an untested hypothesis for this data. [Official model card](https://huggingface.co/Prior-Labs/tabpfn_3_5).

This is a recovery of unfinished experiments. Neither model is assumed to beat V9.

## Stage 1: resolve engineering before reading accuracy results

Allow at most two hours for audit, setup and both pilots. Use an isolated V14 environment and free T4 only. Keep original V9/V13 sources and environments unchanged. Freeze exact package versions after the engineering pilot, before development scores are read.

For TabPFN-3.5:

1. Capture the full traceback, imported module paths and source hashes, package versions, checkpoint revision, size, SHA256 and actual loader path. The saved V11 message alone does not establish the root cause.
2. Use the official version constructor with `ModelVersion.V3_5`. Verify the artifact is the real `tabpfn-v3.5-20260909.safetensors`, preserving its original suffix when copying from the download cache. Check the file format and metadata before fitting.
3. Check the installed loader against the official release implementation. Its `Checkpoint` class distinguishes safetensors from torch checkpoints by suffix and uses a safetensors reader. Fix the demonstrated path/version/format mismatch if found; do not assume changing `weights_only` solves it. [Official checkpoint loader](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/checkpoint.py).
4. Verify the fitted architecture/version and checkpoint hash. Any silent substitution to 2.5 or another model is a failed 3.5 pilot.
5. Official downloads may require account authentication and prior license acceptance. Use existing authorized access if available; record any unmet access requirement. [Official local-download instructions](https://docs.priorlabs.ai/models/accessing-model-weights).

For TabICL, preserve the V11 checkpoint and parameters initially. Test the actual fold size, roughly 2,794 training and 699 validation rows, including unseen/missing categories. Measure inference differences, fold time and peak memory. Use 8 estimators, ensemble batch size one, explicit float32 without AMP/FA3, seed 10242, and the existing fixed 500-row adapter. Preserve documented normalization and feature-shuffle settings from V11; record every resolved argument.

**Proposed numerical contract change, declared before any V14 development:**

- Feature construction must remain exactly invariant to target and ID changes. IDs and validation labels cannot enter the model inputs.
- Same input, seed, pinned environment, inference layout and fresh fit must reproduce within INR 0.01. A new submission notebook must also reproduce its exported CSV within INR 0.01.
- For reordered rows, split batches and individual-row probes only, permit a maximum absolute prediction difference of **INR 10**, with an ordinary RMSE change of at most **INR 1** on the labeled pilot. This separates float32 operation-order noise from the existing same-layout reproduction requirement. Log both differences in INR; do not round outputs to pass.
- Run full-fold permutation and split-batch checks plus at least twelve individual-row probes. Larger changes, changed qualification decisions, nonfinite outputs or leakage fail the pilot. Do not enlarge the tolerance after observing V14 results.
- The original V9 reproduction requirement stays INR 0.01. The proposed INR 10 behavior tolerance is not a measured guarantee that TabICL will pass.

Admit a family solely on access, correctness, numerical behavior, memory and measured runtime. If one fails, test only the other. If both fail, close V14 with the diagnosis and retain V9. No accuracy-selected replacement family.

## Stage 2: two fixed model settings and sixteen comparisons

| Setting | Model | Estimators | Features | Target/output |
|---|---|---:|---|---|
| I8 | Official TabICLv2 regressor, `tabicl-regressor-v2-20260212.ckpt` | 8 | Frozen V9 forty-column builder | Raw INR millions; predictive mean |
| P35_8 | Official TabPFN-3.5 Base, `tabpfn-v3.5-20260909.safetensors` | 8 | Same frozen V9 builder | Raw INR millions; predictive mean |

Use train-only category vocabularies with the documented categorical interfaces, missing/unseen handling and original row/feature order. No new feature search, calibration or target treatment. Fit on all original training-subset labels; score all original validation labels in INR. Apply only the predeclared zero prediction floor if needed.

Evaluate two routes using the same saved component predictions, with `w = 0.25, 0.50, 0.75, 1.00`:

```
V9 = 0.5 * V7 + 0.5 * T9

A: add to full V9
   (1 - w) * V9 + w * M

B: preserve V7, replace within the TabPFN half
   0.5 * V7 + 0.5 * ((1 - w) * T9 + w * M)
```

Here T9 is the original V9 TabPFN-v2 component and M is I8 or P35_8. Route B retains the strong original V7 share. This gives two backbones × two routes × four weights = **16 selectable configurations**, using at most **20 new development fold fits** on five-fold assignments 42 and 71. The routes do not require separate component fitting. No I8/P35_8 cross-blend, estimator refinement or additional family is added after results.

## Stage 3: advance one meaningful candidate

Require positive full-target RMSE gain on both development assignments and mean relative gain at least **0.20%**, about INR 39,500 at the reviewed local baseline. This is a practical threshold, not a significance level or a predicted portal gain.

Rank eligible configurations by descending mean relative gain, then lower effective new-model share (w for A, w/2 for B), then setting name and route. Freeze one candidate, including its features, parameters, seed, route, weight, checkpoint and inference adapter. If none qualifies, stop. Do not lower the accuracy threshold because earlier rounds failed.

Report every configuration, component-alone scores, folds and <=40M / 40–80M / >80M segments. Include error correlation, top-row contributions and the gain after excluding the five largest *baseline* error rows as sensitivity diagnostics only. Neither segment metrics nor exclusion diagnostics replace the full-label selection criterion.

## Stage 4: frozen confirmation

Evaluate the single frozen candidate on random assignments 48211/48212/48213, stratified 27421, and the inherited canonical-city grouped assignment. This requires at most **25 new component fold fits**.

Reuse V13's 35 validated baseline folds only after verifying original inputs, row order, fold assignments, source/configuration/platform provenance and prediction hashes. Keep the original-Mac V7 boundary; do not substitute an unmatched Linux V7 refit. Refit invalid controls only if the measured budget permits; otherwise stop.

Require all three random gains positive, mean random relative gain >=0.20%, lower 95% paired-row gain bound above zero (5,000 draws, seed 8319; resample the same restaurant indices across assignments), and city/stratified deterioration <=1% each. Report all failures. Freeze the candidate before these results and do not tune from them.

All labels have been examined in previous rounds. These confirmations and intervals are conditional robustness checks, not an untouched test or an unbiased estimate of the entire historical search. Changing random seeds would not create new labels. [scikit-learn selection-bias explanation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## Stage 5: reproduction and submission decision

For a qualified candidate, execute a separate standalone notebook that freshly fits required TabPFN/TabICL components, uses the independently verified original-Mac V7 fitted state with that limitation stated, and reproduces the exact CSV within INR 0.01. Expected predictions are verification witnesses read after inference, never inference inputs.

Check exactly 500 original test IDs/order, two exact headers `Registration Number,Annual Turnover`, finite nonnegative INR predictions, CSV round-trip, source/input/runtime/checkpoint hashes and rejection of stale/incomplete/corrupt checkpoints. Save complete folds and their provenance during execution. Release only the owned runtime after collecting artifacts.

After every gate passes, recommend at most **one manual portal trial**. Record its returned RMSE against the exact CSV hash and retain the lowest observed submission. Keep the one-time coding upload pending the final choice. No new local result is an assumed portal score or rank.

If no candidate qualifies or cannot reproduce, deliver the diagnosis and preserve the byte-identical scored V9 CSV/notebook; do not relabel V9 as a new V14 model.

## Proposed compute and deadline

| Work | Maximum allocation |
|---|---:|
| Audit, loader repair, setup and two engineering pilots | 2.0 hours |
| Two development settings | 1.5 hours |
| One frozen confirmation | 2.5 hours |
| Fresh notebook execution, report and delivery | 2.0 hours |
| Total | **8.0 hours** |

This is a proposed separate V14 cap, not a completion estimate, paid-compute authorization or reset of any earlier ledger. At pilot end, forecast the remaining fits using measured times plus 30% contingency and preserve the two-hour delivery reserve. Exclude an infeasible family before reading its development scores; retain its engineering evidence. If the remaining mandatory work will not fit, stop with V9. Persist global elapsed charges across interruptions and bound recovery attempts inside the cap.

Use IST for the working schedule: stop training by October 29, 18:00; finish artifacts by 21:00; target the manual trial by 22:00. The historical portal deadline is October 29 at 11:59 PM with an unconfirmed timezone. Verify it before starting the run and before any upload; an earlier actual deadline overrides these working times.

## Work excluded from this round

Do not repeat V13 caps or square-root treatments, V12 estimator scaling, V6 domain weights, V7 band/tier/residual grids, V8 Ridge/target-encoding grids, or V10 feature cleanup. Do not add a public competition solution's predictions or target matches.

Fine-tuning is a later option only if a backbone first shows useful validation performance and a separately specified inner-validation procedure fits the available hardware. The current official regressor fine-tuning example recommends an 80 GB CUDA GPU; that does not establish impossibility on T4, but it does not justify assuming a free-T4 eight-hour run will support it. [Official fine-tuning example](https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_regressor.py).

The intended result is a completed, reproducible test of two genuinely unmeasured backbones. The plan contains no promised rank or RMSE reduction.
