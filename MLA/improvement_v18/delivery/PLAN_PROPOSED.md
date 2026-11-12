# V18 proposal: task adaptation, a new regression model, and stronger confirmation

Prepared October 29, 2026, IST. **Planning complete; V18 training has not started.** This document proposes an **18-hour elapsed cap on free compute**, compared with V17's six-hour cap. It does not start an execution clock or change any closed experiment ledger.

**Recommendation:** adapt V14's successful TabPFN-3.5 backbone with two fixed context sizes, test EXAONE Tabular, and evaluate two predeclared equal mixtures of those components. Preserve scored V14 until a complete candidate passes every gate and a manual portal trial returns a better score.

The user confirmed that the competition deadline has been extended. The exact replacement date, time and timezone are still pending. The old October 29 working cutoffs must not be treated as the new deadline. Execution needs the actual new deadline before allocating its full time window.

## What V17 established

| Evidence | Measured finding | V18 consequence |
|---|---|---|
| Best observed portal submission | V14, RMSE **11,736,110.7877709** | Keep its exact scored CSV and matching executed notebook. |
| Nori 1.5 | Best blend: 50% Nori + 50% V14; mean local gain **0.096332%** | Below the 0.20% floor. Do not use it as a new baseline or repeat a neighboring-weight grid. |
| Nori/V14 residual correlation | **0.99577 / 0.99581** on assignments 42 / 71 | Its errors were highly similar to V14's; stronger adaptation or another prior has higher priority. |
| Nori sensitivity | Removing the five largest baseline errors changes its 50% blend gain to +10,153.73 / **−1,834.76** INR | The improvement was not uniformly broad. Keep this diagnostic visible; never delete those labels from the primary metric. |
| TabPFN fine-tuning | Passed fresh-fit, reload, permutation, split-batch, individual-row and full-data-size checks; peak allocated GPU memory **4.85 GiB** | T4 feasibility was demonstrated for the specific 512-row recipe. It received **no outer-fold accuracy test**. |
| Why adaptation was excluded | Conservative confirmation forecast **4.61 hours**, versus a 1.5-hour confirmation allowance | Increase and share the research time allocation. V17 did not exhaust its six-hour total cap; it finished in about 55 minutes. |
| Fine-tuning learning curve | The pilot selected epoch 3 after 11 inner epochs | More epochs alone are not the proposed improvement. Use slower, controlled adaptation with inner stopping. |
| Xiaomi | Reordering rows changed predictions by up to **2,290.73 INR**, above the 10 INR limit | Exclude the unchanged recipe. No accuracy conclusion was established. |

The [read-only reconstruction](review_v17.py) independently verified twenty hashed fold outputs: ten Nori outputs and ten original P35 control outputs. It reconstructed all five blends and checked input hashes, fold membership, training-target hashes and scored V14 files. See [review_evidence.json](review_evidence.json). No V18 model was fitted during this review.

The 48 labels above 80M still account for approximately 68% of baseline squared error. This does not establish label mistakes or the hidden-test target distribution. Keep all 3,493 labels, original-scale RMSE, and the original 500 test IDs/order.

## Three fixed component recipes

Use V14's unchanged forty-column builder and V17's train-fitted categorical codec. Explicitly identify all eight categorical columns, including Cuisine 1 and Cuisine 2. Remove target and registration ID before feature construction; preserve missing/unseen values and the existing city/header corrections. Fit targets in INR millions and convert outputs back to INR. Apply only a fixed zero floor; no target caps, label removal or output rounding.

| Priority | ID | Fixed hypothesis |
|---:|---|---|
| 1 | **FT512** | Learn task-specific weights in the proven T4-sized TabPFN-3.5 adaptation path. |
| 2 | **EXA8** | Test the released EXAONE regression checkpoint with eight uniform members. |
| 3 | **FT1024** | Test whether a larger training context improves adaptation, with the other FT settings identical to FT512. |

### FT512 and FT1024

Use the exact V14 TabPFN-3.5 checkpoint, SHA256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`, explicitly selecting `ModelVersion.V3_5`. Retain the proven `tabpfn==9.2.0` environment and verify the installed fine-tuning source against the reviewed published release. Do not upgrade V14's control environment.

Both recipes use a **constant learning rate of 3e-6**, AdamW weight decay 0.01, maximum 30 inner-selection epochs, patience eight, gradient clip one, activation checkpointing, query fraction 0.2 and seed 10242. Use one estimator consistently for training, validation and final inference, fixed preprocessing seeds, full training context during inference, and the official CRPS + predictive-mean MSE losses with weights 1/1 and no loss clipping. Training uses the wrapper's CUDA AMP; inference uses float32. Only the context-plus-query chunk size differs: **512 versus 1,024**.

Explicitly set `use_lr_scheduler=False` in selection and refit. The published 9.2.0 source derives cosine/warmup duration from the requested epoch count. Selecting an epoch in a 30-epoch run and refitting for that shorter count therefore changes the schedule. A constant rate removes that particular discrepancy; it does not make two different training subsets equivalent. The lower rate is a new fixed hypothesis motivated by the pilot's early overfitting, not an established improvement. These are revised recipes, not a claim to have completed V17's exact unscored recipe. [Official fine-tuning implementation](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_base.py).

Use an inner 15% holdout, seed 9091 + outer-fold index. For city validation, split whole canonical cities. Fit inner preprocessing only on inner training rows. Select the lowest inner raw-scale MSE among epoch zero and completed epochs, choosing the earlier epoch on ties. Start again from the original checkpoint, refit preprocessing on all outer-training rows and train for the selected count without validation stopping. Apply the same procedure to final full-data fitting. Outer-validation labels never choose epochs or preprocessing.

Report the frequency of epoch-zero selection, convergence, actual precision, optimizer steps and selected epochs. Include a matched unadapted **one-estimator** control on every FT evaluation, shared across the two recipes where its input/runtime context is identical. It is diagnostic only and cannot become another selectable blend. This separates learned adaptation from merely changing V14's eight-estimator layout. The official wrapper exposes the relevant controls and losses. [Regressor source](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_regressor.py).

The old 512-row pilot does not establish 1,024-row memory, accuracy or repeatability. Both revised recipes require fresh engineering admission.

### EXA8

Pin official source revision **`d149d0f0441053f0cf6b2b45f4b2067a81619cf9`**. Use the released regression weights, not the classifier. Public Hub metadata identifies revision **`093ad1c2613e1e95d26df8610451d2e02ed53cbe`**, file `exaone-tabular-regressor-v1_default.safetensors`, 84,502,160 bytes, advertised LFS SHA256 **`24891e14e649c6509f07ee62b5583e06544b68eaf9a9195ee58777eb9cf6fa0d`**. Verify actual downloaded bytes during execution; metadata alone is not a completed download. [Official model card](https://huggingface.co/LG-AI-Research/EXAONE-Tabular).

Freeze eight members, seed 10242, CUDA float32, `point_estimate='mean'`, `member_weighting='uniform'`, `n_svd=0`, `svd_gate=False`, `svd_split=False`, support-row limit 100,000, feature limit 500 and query batch limit 500. Use the public immutable manifest/loader and explicit Boolean categorical mask. No internal NNLS weighting, member trimming, median readout, extra SVD arm or context-row subsampling. The pinned source supports these overrides; cached web documentation showed older defaults, so defaults must not be inferred from the model card. [Pinned config](https://github.com/LGAI-Research/EXAONE-Tabular/blob/d149d0f0441053f0cf6b2b45f4b2067a81619cf9/src/exaonetabular/config.py), [pinned regressor](https://github.com/LGAI-Research/EXAONE-Tabular/blob/d149d0f0441053f0cf6b2b45f4b2067a81619cf9/src/exaonetabular/regressor.py).

Use its supported memory planner with a budget no greater than 12 GiB or currently free GPU memory. Check that all intended support rows and members remain present, fitted transforms are reused, and query batches do not refit statistics. T4 feasibility, saved-state persistence and numerical tolerances remain execution gates. Author benchmarks motivate the test; they do not establish a gain here.

## Engineering admission and time accounting

Before scoring, verify original data/source/control hashes and the V14 CSV/notebook. Smoke-test actual entrypoints, interrupted/stale/corrupt cache rejection and the no-finalist path locally. Reuse V17's compatible dependency pins where possible; EXAONE's dependency ranges permit that starting point. Record the complete resolved environment and source/checkpoint hashes before development.

Run pilots in priority order FT512, EXA8, FT1024. Each must exercise an actual outer-sized problem (about 2,794 training rows / 699 queries), a 3,493-row full-data-size probe, fitted-state reload and a repeat fit. Verify target/ID exclusion, train-only preprocessing, missing/unseen values, all intended rows/members, and a fixed row after changing other query rows.

Retain the established tolerances: same-layout fresh/reloaded predictions and final CSV reproduction **<0.01 INR**; permutation, split-batch and at least twelve individual probes **≤10 INR**; labeled pilot RMSE movement **≤1 INR**. No rounding or tolerance relaxation to admit a candidate.

Record fixed setup/loading costs separately from training-epoch, validation-forward, refit and final-query costs. Forecast the maximum 30 selection + 30 refit epochs, using observed timings and **30% contingency**, including the matched unadapted controls. Include fresh random-split control reconstruction, full-data fits and a separate fresh notebook execution. V17's 6.46-hour development-plus-confirmation estimate is historical context, not a timing promise for the revised recipes.

Freeze the admitted set using correctness and complete-workload forecasts only. If the three recipes cannot fit, remove FT1024 first, then EXA8; preserve FT512's full evaluation if feasible. Report any exclusion explicitly. Do not substitute an accuracy-selected fourth model. Preferred stage allocations can borrow from the shared research pool; do not repeat V17's exclusion solely because a preferred confirmation slot is short. The total cap and delivery reserve remain binding.

## Fixed development and combinations

Use the existing five-fold development assignments **42 and 71**, identical to verified V14 controls. Complete both assignments for every admitted component before selection. The selectable routes are exactly:

1. FT512 alone.
2. EXA8 alone.
3. FT1024 alone.
4. Equal mean of FT512 and EXA8.
5. Equal mean of FT1024 and EXA8.

A pair exists only if both components were admitted. No FT512/FT1024 mixture, Nori refinement, stacking fit or row routing is added.

For each route M, evaluate exactly:

```text
B = V14 = 0.25*original_V9 + 0.75*original_P35_8
candidate = (1-w)*B + w*M
w = 0.10, 0.25, 0.50, 0.75, 1.00
```

Maximum: **30 new component outer evaluations, ten matched one-estimator control evaluations, and 25 selectable blends**. The twenty FT component evaluations each include inner selection plus a fresh outer-training refit: up to forty FT training runs before confirmation, excluding pilots. Pair routes reuse predictions and require no extra development training. All recipe/weight choices count toward the reported search.

Advance at most one route/weight. Require positive full-label RMSE gain on both assignments and mean relative gain **≥0.20%**. Rank by descending mean relative gain, then lower new-component weight, fewer components, then route ID. Lock its complete recipe, component weights and notebook protocol before confirmation. Never adjust weights using confirmation or portal results.

Report standalone component errors, residual correlations, fold consistency, <=40M / 40–80M / >80M errors, missing/unseen-city errors, largest row contributions, and sensitivity excluding the five largest baseline errors. These are diagnostics; true turnover bands never choose inference-time models. No segment result changes the predeclared main gate.

## Confirmation on new random assignments

Use new five-fold random assignments **918601, 918602 and 918603**, plus the existing stratified assignment 27421 and inherited canonical-city groups. Predeclare them now and keep their scores out of model/weight selection.

Construct V14 controls for the new random assignments from its frozen recipe: original-platform V7 on the Mac, pinned T9 and P35_8 on the T4. First reproduce one existing control fold on its proper platform. Do not substitute Linux V7 fits for the original-Mac control or reuse a full-data model to predict validation rows. Cached city/stratified controls require their full provenance checks. Baseline building time is charged and forecast before admission; Mac and GPU work may run concurrently under one coordinator.

Confirm only the locked candidate: 25 new component evaluations if a single component wins, or **50** for an FT + EXA8 pair. The pair contains only one FT recipe, so it needs at most fifty FT training runs, not one hundred. Add 25 matched one-estimator diagnostic evaluations if an FT recipe wins. The three new random assignments need fifteen V7, fifteen T9 and fifteen P35_8 control fits; include all forty-five in the workload accounting.

Qualification requires positive gains on all three new random assignments, mean random relative gain **≥0.20%**, a positive lower 95% paired-row gain bound (5,000 draws, seed 8319; same restaurant draws across assignments), and deterioration no worse than 1% on each city/stratified check. No retuning or second finalist after a failure.

These are new partitions of historically examined labels, not an untouched test set. The bootstrap interval is conditional robustness evidence and does not include all selection, retraining or hidden-test shift uncertainty.

## Proposed larger resource envelope

**Maximum 18 elapsed hours**, including implementation, setup, downloads, failures, interruptions and recovery. Free T4 plus the existing Mac only. No paid GPU or guaranteed uninterrupted Colab session.

| Work | Preferred allocation |
|---|---:|
| Audit, implementation, setup and engineering | 2 h |
| Fixed development, including matched controls | 5 h |
| One frozen confirmation, including new V14 controls | 7 h |
| Bounded recovery / research contingency | 1 h |
| Fresh standalone notebook, verification and delivery | **3 h reserved** |
| Total | **18 h** |

The first four rows share a **15-hour research ceiling**; preferred allocations are not separate stop rules. Reserve the final three hours before admitting expensive work. End early when the declared experiments finish. Checkpoint every complete fold and, through supported APIs, completed inner epochs/optimizer state. Charge recomputation and interruptions; resume only matching sources, inputs, rows, targets, runtime, configuration and hashes. Allow at most two owned-runtime recovery allocations inside the cap, without account switching or paid fallback to evade resource limits.

Let D be the newly confirmed competition deadline and T the authorized execution start. Effective total time is `min(18 hours, D - T - 1 hour)`. Training must stop at least **four hours before D**; final artifacts must be ready at least **one hour before D**. If the complete workflow cannot fit those limits, reduce the admitted set before accuracy scores or retain V14. No dated run schedule is claimed until D is known. Prior V17 charges remain closed and unchanged.

## Delivery and score decision

For a qualified candidate, independently execute a standalone notebook that freshly fits its FT/EXAONE components and required TabPFN controls and reproduces the final CSV within 0.01 INR. Preserve the disclosed verified original-Mac V7 fitted-state boundary. Expected predictions are witnesses checked after inference, never model inputs.

Verify exactly 500 original registration IDs/order, exact two headers, finite nonnegative INR predictions, CSV round-trip, all hashes, every complete fold, every selectable score, decision reconstruction, source revisions and inner-selection records. Collect full logs, release only owned compute, and verify the selected account's runtime state.

Recommend at most one **manual** portal trial after all gates pass. Record its actual score against the exact CSV hash and retain whichever submission has the lowest observed portal RMSE. Keep the one-time coding upload tied to that final choice. If no candidate qualifies or reproduces, preserve byte-identical V14 files and report the failure; do not export an unqualified V18 submission.

A **1% local reduction** is a useful ambition, about 196,495 INR at the current development baseline. It is not a forecast or a promised portal/rank gain. The executable objective remains a verified improvement over V14 that survives the complete qualification process.

Research snapshots, exact public release metadata and the reasons for deferring other model families are in [RESEARCH_SOURCES.md](RESEARCH_SOURCES.md). This proposal does not allocate a runtime or start training.
