# V17 proposal: adapt the winning model and test two new regression priors

Prepared October 29, 2026, IST. **Research and planning completed; training has not started.** No new runtime, checkpoint download, submission CSV, or portal trial was created for V17.

**Recommendation:** retain scored V14 as the control; test Xiaomi-TabLDM and Nori 1.5, plus one conditional low-memory fine-tuning recipe for V14's TabPFN-3.5. This changes the model's prior or learns task-specific weights. Another temperature, feature-view, estimator-count, target-cap, or small TabM blending grid has lower priority after the measured history.

The objective is a materially better model. A mean local reduction of 1% would be a substantial result, approximately INR 196,495 at today's development baseline. That is an ambition, not a forecast or a substitute for the existing 0.20% qualification floor. Local gains do not predict the same percentage gain on the portal.

## 1. Evidence behind the change

| Observation | Verified result | Decision |
|---|---|---|
| Best observed portal file | V14, RMSE **11,736,110.7877709** | Preserve its exact CSV and matching executed notebook. |
| Successful V14 change | 25% V9 + 75% actual TabPFN-3.5 Base, eight estimators | The strongest local precedent supports a new pretrained backbone or adapting this successful one. |
| V15 refinement | Best development gain 0.022838%; failed 0.20% floor | Deprioritize more nearby temperatures and feature views. |
| V16 TabM | Best gain **0.031839%**, at 10% TabM | No qualified finalist; no V16 portal score. |
| TabM alone | RMSE 19.896M / 19.955M on assignments 42 / 71 | This specific TabM recipe was weaker than V14. |
| TabM/V14 error correlation | 0.98447 / 0.97790 | Some diversity, insufficient improvement at the tested weights. |
| V14 error concentration | 48 labels above 80M contribute **68.10–68.14%** of squared error; five rows contribute **34.84%** | Study where candidate improvements come from; retain all labels. |
| Mitra-v2 | Excluded before accuracy testing: public prediction wrapper discards fitted-child state | No Mitra accuracy conclusion; do not repeat its unchanged lifecycle pilot. |

The [read-only audit](review_local.py) reconstructed all five V16 comparisons from ten original P35 and ten TabM fold outputs, verified their file hashes, training/validation membership and training-target hashes, and verified the scored V14 CSV/notebook hashes. [Machine-readable evidence](review_evidence.json).

The audit confirms 3,493 training rows, 500 test rows, 32 original predictors and a 400M maximum label. Error concentration does **not** establish wrong labels, hidden-test composition, or a reason to remove extreme restaurants. True turnover bands are diagnostics and cannot choose a model at inference.

Older work already covered domain weights, band/tier specialists, nested residual trees, Ridge, HGB/target encoding, EBM, feature cleanup, RealMLP, larger TabPFN ensembles, caps, square-root distribution moments and TabICLv2. Repeating those as if untested would overstate V17's novelty. The reviewed source inventory and alternatives are in [RESEARCH_SOURCES.md](RESEARCH_SOURCES.md).

## 2. Three fixed experiments

All settings use V14's unchanged forty-column builder, original training labels in INR millions, six explicitly identified categoricals, train-only fitted preprocessing and predictive means. Target/ID columns never enter features. Preserve the endorsed-header correction and existing city aliases. Every training row remains available to each outer-training algorithm. The only output bound is a fixed zero floor.

### XIA32: Xiaomi-TabLDM, base regression path

Use the official `TabLDMRegressor` and `occams/Xiaomi-TabLDM` regression checkpoint, including the September regression updates. The authors describe synthetic pretraining and regression support. Its different architecture/prior is the hypothesis; benchmark standings are not evidence for these restaurant labels. [Official repository](https://github.com/xiaomi-research/xiaomi-tabldm).

Proposed configuration: 32 estimators, `norm_methods=['none','power']`, random feature permutations, seed 10242, ensemble batch size one, `kv_cache=False`, `use_amp=False`, `use_fa3=False`, CUDA float32, documented offloading only if required. Use the decoded distribution mean. Set `enhance_candidates=False` and disable the high-kurtosis target ensemble. These settings are supported by the reviewed regressor source. [Official regressor implementation](https://github.com/xiaomi-research/xiaomi-tabldm/blob/main/tabldm/_sklearn/regressor.py).

The current source defaults to enhanced candidate selection. Explicitly selecting the base path prevents uncounted internal searches, transformed-target members and fitted blend weights. Record feature-only outlier handling separately from labels; verify the complete original target vector reaches fitting unchanged. No accuracy-driven reduction from 32 members to a cheaper recipe after screening.

### NOR15: Nori 1.5, four-member regression recipe

Use the released `model='nori-1-5'` checkpoint, approximately 21M parameters, and its four-member inference recipe. It is described as synthetic-pretrained and regression-specific. Its quantile head supplies a different distributional prior. Use the continuous distribution mean; no discretization or snapping. Existing measurements do not establish it as better than TabPFN-3.5. [Official Nori repository](https://github.com/Synthefy/synthefy-nori).

Explicitly disable optional Yeo–Johnson target augmentation with `augmentations=()`, set `quantile_collapse='mean'`, and declare the six categorical columns. Disable text/ID encoders, large-context policy searches and target snapping. Set an explicit seed through the supported lower-level configuration and log it. These wrapper controls are documented in the [official API source](https://github.com/Synthefy/synthefy-nori/blob/main/src/synthefy_nori/api.py).

Source admission must inspect the *actual four-member checkpoint config*. Some current package preprocessing can use query features, and memory fallbacks can drop context rows. Require training-only fitting, `fit_on_test=False` wherever supported, no context-row dropping and no degraded-pipeline fallback. Freeze any supported configuration overrides before scores. If a required train-only or T4-compatible path is unavailable, exclude NOR15 without patching private internals. The web reader could not retrieve the Nori 1.5 model card directly; the repository's advertised availability still needs a download/access pilot. [Official inference documentation](https://github.com/Synthefy/synthefy-nori/blob/main/docs/inference.md).

### P35FT512: fine-tune V14's successful backbone, conditional

Use the exact original TabPFN-3.5 checkpoint hash `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`. Explicitly select `ModelVersion.V3_5`; never rely on a package default or substitute 2.5/3/Fast. Isolate newer fine-tuning dependencies from the original V14 control environment.

This is a proposed small-memory experiment: maximum 30 epochs, context-plus-query chunks of 512 rows, query fraction 0.2, AdamW learning rate 1e-5 / weight decay 0.01, gradient clip 1, activation checkpointing, one estimator during training/validation/final inference, fixed preprocessing seed, and seed 10242. Keep all three estimator counts equal to honor the wrapper's fixed-preprocessing constraint. Use the official CRPS-plus-predictive-mean-MSE loss, weights 1/1; no target or MSE clipping. Record actual mixed precision and attention behavior. These choices adapt the model; they do not replicate the original eight-estimator component. [Official fine-tuning regressor](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_regressor.py).

Select epochs using only a 15% inner holdout, seed 9091 + outer-fold index, patience eight. Compare initial weights and all completed epochs using inner raw-scale MSE. Start again from the original checkpoint and fine-tune on the entire outer-training subset for the selected count, without validation-based stopping. Include initial-weights selection explicitly as epoch zero; report its frequency. Refit preprocessing within the corresponding subset. For city validation, the inner holdout must contain disjoint whole cities. Use the same selection/refit procedure for final full-data training.

The official loop automatically uses CUDA AMP/GradScaler; a float32 inference setting does not turn training into float32. Its fitted lifecycle and precision must pass fresh-fit and inference checks. The wrapper's saved checkpoint, selected-epoch metadata and all-data refit must be supported interfaces. [Official fine-tuning base implementation](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_base.py).

**T4 feasibility is unproven.** The official example recommends an 80 GB CUDA GPU. Small chunks, one estimator and activation checkpointing justify a short engineering pilot, not an assumption that 16 GB is enough. No private training-loop patch, BF16-only kernel, paid GPU, silently shortened run or larger-context search is proposed. [Official example and hardware recommendation](https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_regressor.py).

As a diagnostic, compare original weights under the fine-tuning environment with original weights under the original V14 environment. Log dependency-induced differences. A selected epoch zero is not evidence that learning improved the backbone. The matched unadapted model is not a fourth selectable setting.

## 3. Engineering admission before development scores

First preserve V14's CSV SHA256 `39d49f79beb8549d04ffc78d134aad2407dc5e73914dc3c30e8bb162a3c0ba05` and notebook SHA256 `86fae4ea97d6fd35c2ed485c60f26f02c98fccb3cbeda9a10228cc48e788318e`. Verify the inherited control registry, original inputs, sources, fold assignments and original-Mac V7 provenance. A current package update cannot silently regenerate the control.

Use one shared hour for audit/setup and engineering, with suggested ceilings of 10 minutes audit, 15 XIA32, 15 NOR15 and 20 P35FT512. The shared ceiling takes precedence. Audit source defaults and available checkpoints before heavy installs. Admission depends on access, correctness, supported persistence, actual memory and complete-workload timing; never on development RMSE.

For every setting:

1. Exercise an actual outer-fold-sized problem: about 2,794 context/training rows and 699 queries. Include a 3,493-row full-data-size memory probe. FT timing includes epoch selection and a fresh refit, with maximum-epoch extrapolation.
2. Check exact target/ID feature invariance, fold-local vocabularies/statistics and explicit missing/unseen categories. Change other query rows and check a fixed row's prediction; fitted preprocessing cannot depend on the current query batch.
3. Require fresh same-layout fits and saved/reloaded predictions within INR 0.01. Full-fold permutation and split-batch probes plus at least 12 individual rows must differ by at most INR 10, with labeled-pilot RMSE movement at most INR 1. No rounding to pass. Record actual precision, seeds and kernels.
4. Verify all intended context rows and ensemble members were used, no hidden candidate selection, no fallback checkpoint, and no stale cache. Validate persistent fitted-model inference without retraining for each probe.
5. Freeze exact revisions, checkpoint SHA256, package versions, resolved parameters, preprocessing config and inference layout before development. A failed candidate is excluded; do not replace it with another accuracy-selected family.

Pilot order is XIA32, NOR15, P35FT512. Let measured complete outer-evaluation costs be t(X), t(N), t(F), including fit plus all queries and FT inner selection/refit. Forecast remaining development as ten evaluations per admitted setting, confirmation as 25 evaluations of the **slowest admitted setting**, full-data fitting plus a separate fresh notebook run, verified controls and reports. Add 30% contingency. Protect the two-hour delivery reserve. Exclude unaffordable settings before screening in reverse priority order, without using their scores. If none fits, close V17 and retain V14.

## 4. Fixed development and one finalist

Reuse five-fold assignments 42 and 71. For each admitted component M evaluate exactly:

```text
candidate = (1 - w) * V14 + w * M
w = 0.10, 0.25, 0.50, 0.75, 1.00
V14 = 0.25 * original V9 + 0.75 * original P35_8
```

At most **30 new outer evaluations and 15 selectable blends**. FT evaluations contain an inner selection run plus a fresh all-outer-training refit, so maximum development work is 20 ICL evaluations plus 20 fine-tuning runs, besides controls/pilots. Repeated probes must not refit models. No cross-family stacking, row routing, extra feature settings or neighboring-weight search.

Advance only a candidate with positive full-label RMSE gains on both assignments and mean relative gain at least **0.20%**. Rank by descending mean relative gain, then lower new-component weight, then setting ID. Freeze at most one candidate before confirmation. Report every setting and blend, including standalone model RMSE, residual correlations, the <=40M / 40–80M / >80M bands, missing/unseen-city errors and row-level squared-error contributions.

Report improvement after excluding the five largest baseline errors as a sensitivity diagnostic only. A >80M improvement is informative but not compulsory: earlier successful versions sometimes traded off segment accuracy. If no candidate qualifies, preserve V14 and do not manufacture a new submission.

## 5. Frozen confirmation and delivery

Confirm only the locked candidate on random five-fold assignments 48211, 48212, 48213, stratified 27421 and inherited canonical-city groups. Reuse controls only after exact provenance checks. This is at most 25 more component evaluations, or 50 FT runs if P35FT512 advances. City-safe inner epoch selection is required for FT; no outer validation labels enter training or checkpoint selection.

Qualification retains the existing rules: positive gain on all three random assignments, mean random relative gain >=0.20%, positive lower 95% paired-row gain bound using 5,000 draws/seed 8319 with identical restaurant draws across assignments, and at most 1% deterioration on each city/stratified check. No retuning from confirmation. Historically examined labels and reused splits provide conditional robustness evidence; they do not create an untouched test.

A qualified candidate needs a freshly executed standalone notebook that refits the new component and required TabPFN controls and reproduces its exported CSV within INR 0.01. Retain the disclosed original-Mac V7 fitted-state boundary. Check exactly 500 original IDs/order, exact `Registration Number,Annual Turnover` headers, finite nonnegative INR predictions, CSV round-trip and all hashes. Save complete folds, all comparisons, inner selections, engineering exclusions, runtime ledger and decision reconstruction. Expected predictions are verification witnesses, never inference inputs.

## 6. Proposed resource cap and deadline

Propose **six elapsed hours maximum on free T4**, including installation, downloads, interruptions and recovery. This proposal starts no clock and does not extend/reset the closed V16 ledger.

| Stage | Maximum allocation |
|---|---:|
| Audit, setup and all engineering pilots | 1 h |
| Fixed development | 1.5 h |
| Single frozen finalist confirmation | 1.5 h |
| Fresh notebook, verification and delivery | 2 h |
| Total | 6 h |

The local V16 deadline record states the user previously confirmed **October 29, 2026, 23:59 IST** for the portal. This planning turn did not independently inspect the authenticated portal. Preserve the stricter working cutoffs: stop training by 18:00 IST, finish artifacts by 21:00 IST and leave manual submission time before 22:00 IST. Effective available time is the smaller of the proposed cap and these cutoffs; starting late does not create another six hours. Verify the portal remains open before a manual trial.

After every gate passes, recommend at most one manually uploaded trial, record its actual returned score against the exact CSV hash, and retain the lowest observed submission. The one-time coding upload follows the final model choice and actual portal state. No automatic portal upload or promised rank/RMSE is part of this plan.

**Expected deliverable:** a completed, reproducible test of up to three materially different hypotheses, with V14 preserved whenever the evidence is insufficient. A large gain is possible only if these methods discover usable signal that the existing ensemble misses; internet benchmark wins cannot establish that outcome.
