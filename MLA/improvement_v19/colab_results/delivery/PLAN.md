# V19 plan: test new regression priors against the scored V14

Prepared October 29, 2026, IST. **Research and historical audit complete; V19 training has not started.** The new Colab account was authenticated and independently checked. No runtime was allocated. This is a proposed experiment, not a claim of better predictions.

**Recommendation: pilot Causilo, Kumo Tabular Small and TabDPT v1.3; include Kumo Medium only if the complete workload fits before seeing new accuracy results.** Preserve V14 and compare a small, fixed set of blends against it. The other models in the screenshots are valid research leads, but do not all deserve training time in the same experiment.

## What the project evidence establishes

V14 remains the best observed portal submission: **RMSE 11,736,110.7877709**, from the user-supplied October 28 result. It combines **25% original V9 + 75% TabPFN-3.5 Base, eight estimators**. The historical displayed rank was 9; no live rank was checked.

The [independent audit](history_audit.json) verified 110 hashed historical fold outputs, the original inputs, training-target and fold membership hashes, and all 40 V15–V18 blend comparisons. The reconstructed metrics match the saved tables exactly in the audit runtime. The interrupted original V15 attempt is distinct from its completed retry.

| Completed experiment | Best mean relative development gain over V14 | Result |
|---|---:|---|
| V15 retry: native features/temperature | 0.022838% | Below the existing 0.20% qualification floor |
| V16: TabM | 0.031839% | Below the floor |
| V17: Nori 1.5 | 0.096332% | Below the floor |
| V18: TabPFN fine-tuning | 0.039920% | Below the floor; worsened assignment 42 |

V14's development RMSE is 19,649,033.56 / 19,649,905.45 on assignments 42 / 71. **Local RMSE and portal RMSE measure different labeled rows and cannot be interchanged.** None of V15–V18 has a new portal score.

Only 48 of 3,493 training targets exceed INR 80M, yet they account for approximately **68.1% of V14's development squared error**; the five largest errors account for approximately **34.8%**. This explains why a small average gain can change sign across splits. It does not establish incorrect labels or the hidden-test distribution. Keep every original target and evaluate original-scale RMSE.

## Review of the screenshot models

All seven named model families have an official regression implementation or checkpoint. “Supports regression,” “fits a T4,” and “improves V14” are separate claims. Only the first has been verified here for the new candidates.

| Model | Verified finding and practical consequence | V19 decision |
|---|---|---|
| [Causilo](https://github.com/nums-ai/causilo) | Current package 1.0.3 requires **PyTorch >=2.13**. Authors publish strong TabArena regression results; their resource CSV reports **7.544 GiB allocated / 10.403 GiB reserved** during regression fit on H100. Those are aggregate fit measurements, not this dataset's T4 peak. The comparison includes TabPFN-3, not the complete V14 ensemble. | First conditional pilot, **C8**. Its new environment must not upgrade V14's control environment. |
| [NVIDIA Kumo Tabular](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.KumoTabular) | Small/Medium/Large support regression. The core model returns **999 quantiles**; the official recipe handles categorical inputs before the numerical core. The constructor defaults to **Large**. | Explicit Small **KS8**; optional predeclared Medium **KM8**. Do not accidentally load Large or interpret the median as an RMSE-oriented mean. |
| [TabDPT](https://github.com/layer6ai-labs/TabDPT-inference) | v1.3 is real; current package is 1.3.1. Its default `use_flash=True` forces the Flash Attention backend. Unlike V18's EXAONE path, it exposes a supported `use_flash=False` switch. | Core pilot **D8**, with Flash-only execution and compilation disabled. A real T4 timing/memory test is still required. |
| [Google TabFM](https://github.com/google-research/tabfm) | Official regression support and both JAX/PyTorch backends exist. Neither backend has been tested on this restaurant task in this review. | Reserve for another bounded campaign; avoid adding a backend/checkpoint integration project to the core search. |
| [LimiX-2](https://github.com/limix-ldm-ai/LimiX) | Official release supports regression; advertised large configuration is about 400M parameters. Current source requires Python >=3.12 and PyTorch >=2.9.1. | Defer the large model. The separate **2M-parameter LimiX-2M** is a useful additional lightweight research lead, not the same model as LimiX-2. |
| [TabSTAR v1.1](https://github.com/alanarazi7/TabSTAR/blob/master/docs/model_cards/tabstar-v1.1-reg.md) | A distinct regression checkpoint exists. Downstream fitting uses LoRA and validation-based epoch averaging; it is not simply another zero-shot call. This dataset has short category labels, not a rich free-text review column. | Lower priority. Text semantics might help, but that has not been demonstrated here and adds inner-selection work. |
| [TabSwift](https://github.com/LAMDA-Tabular/TabSwift) | The repository exposes a regressor and a shared classification/regression checkpoint. Current wrapper uses the older sklearn `_validate_data` interface and does not pin the automatic Hub download. | Worth a later lightweight pilot, with an explicit compatible sklearn version and pinned local checkpoint; “lightweight” alone does not prove current Colab compatibility. |

These priorities are a practical inference from the official implementations and local experiment history, not an external leaderboard ranking. Source commits, package metadata and published checkpoint identities are saved in [the research register](RESEARCH_SOURCES.md).

## Fixed recipes and data handling

Start with V14's unchanged **forty-column feature builder**. Exclude turnover and registration ID before constructing features, retain the existing city/header corrections, and mark all eight categorical columns, including the two split-cuisine columns. Fit category vocabularies, imputers and numerical transformations on outer-training rows only. Preserve missingness and unseen-category handling. A numeric adapter needed by a model is part of its fixed recipe, not an extra selectable feature experiment.

Use original turnover divided by 1,000,000 for model targets, then convert predictions back to INR. Use the same fixed zero floor as the baseline. No logarithmic target arm, target caps, label deletion, learned tail correction, target-band routing or portal-based calibration is added.

| ID | Predeclared configuration |
|---|---|
| **C8** | `causilo==1.0.3`, official `CausiloRegressor`, eight estimators, seed 10242, CUDA, `use_kv_cache=False`, `retain_preprocessing=True`, default **mean** output. Use the pinned regression weights, not the classifier. Keep all outer-training rows. |
| **KS8** | Reviewed NVIDIA source commit, `KumoTabular(task='regression', size='small')`, eight estimators, seed 10242 through the official generator, official frozen preprocessing/output recipe. Use the fitted context API and reuse its transformations. Convert the 999 returned, inverse-transformed quantiles to a fixed arithmetic grid mean in float64. This is a numerical mean approximation, not the exact continuous expectation. No median/quantile-readout search. Retain the official 20% trimmed ensemble aggregation. |
| **D8** | `tabdpt==1.3.1`, `TabDPTRegressor(device='cuda', use_flash=False, compile=False)`, eight prediction ensembles, seed 10242, `context_size=None`, explicit query batch size 256, `output_type='mean'`; retain other reviewed regressor defaults. Train-only category encoding/imputation feeds its numerical interface. All training context rows must remain present. |
| **KM8** | Identical to KS8 except `size='medium'`. Admit or exclude on engineering and total workload before any V19 development accuracy is revealed. |

Pin the whole resolved dependency environment and actual weight bytes, not only a package's headline version. Public Hub metadata was read; model weights were **not** downloaded in planning. The [checkpoint register](checkpoint_review.json) records advertised hashes and exact revisions. Actual downloads must match those hashes during implementation.

V14's recorded control uses PyTorch 2.11.0+cu130 and TabPFN 9.2.0. Causilo requires a separate environment. Keep candidate subprocess environments isolated from the frozen V14 environment; share them only if their complete resolved locks are compatible. A final notebook must install and invoke these explicit environments reproducibly. A new account does not change model quality or guarantee GPU access.

## Engineering admission, before accuracy selection

1. Verify original inputs, V14 references and frozen control provenance. Reproduce an existing control fold on its original compatible platform. Preserve the original-Mac V7 component; Linux retraining is not an equivalent substitute.
2. On the user's selected account, inspect the allocated GPU, free memory, CUDA/driver and Python/package versions. Use one owned free runtime. Do not request paid resources. Account sign-in alone does not prove T4 availability.
3. Pilot C8, KS8, D8, then KM8. Each needs a real outer-sized case (about 2,794 training / 699 query rows), a 3,493-training-row plus 500-query case, finite outputs, fitted-state reload and a fresh repeat. Measure both allocated and reserved GPU peaks and host RAM. Prefer <=12 GiB GPU use with headroom, subject to the actual device.
4. Check target/ID exclusion, train-only transforms, missing/unseen values, row order, query batching, at least twelve individual rows, and a fixed row after replacing other query rows. D8 must actually execute its supported non-Flash path. For Kumo, check inverse target units and quantile aggregation explicitly.
5. Freeze numerical behavior before scoring: V14 reproduction keeps its historical <0.01 INR requirement. For **new candidates**, use `atol=10 INR, rtol=1e-6` for fresh/reloaded and query-layout comparisons, plus <=10 INR labeled-pilot RMSE drift. This is a proposed precision-aware numerical rule: single-precision arithmetic at high turnover can exceed a cent without changing meaningful accuracy. It does **not** lower the 0.20% accuracy requirement. No output rounding, candidate-specific tolerance tuning or relaxed treatment after seeing a failure. Material query-composition dependence still excludes a recipe.
6. Forecast every admitted recipe's full development, slowest possible finalist confirmation, baseline rebuilding, fresh full-data fits and fresh notebook execution, with **30% contingency**. Freeze admission before revealing development scores. If over budget, drop KM8 first, then the slowest remaining recipe needed to fit the complete workflow; break timing ties by lower stated priority. Never select admission using early accuracy.

The source review found no need to repeat V18 fine-tuning, unchanged EXAONE Flash-only execution, Nori neighboring-weight searches, or failed Xiaomi inference behavior. If core pilots fail, report the failure and retain V14; do not silently expand to the reserve models after examining scores.

## Development and one locked finalist

Use the exact existing five-fold assignments **42 and 71**, with verified V14 out-of-fold controls. Finish both assignments for every admitted component. The selectable routes are each admitted component individually, plus **one equal average of all admitted core components among C8/KS8/D8**, provided at least two core components passed. KM8 is not added to that mean because KS8 already represents the Kumo family. Do not choose the mixture's membership from accuracy or correlation.

For each route M, compare exactly:

```text
B = V14 = 0.25*original_V9 + 0.75*original_P35_8
candidate = (1-w)*B + w*M
w = 0.10, 0.25, 0.50, 0.75, 1.00
```

Maximum: **40 new development fold outputs and 25 route/weight comparisons**. Mixtures reuse saved predictions. Keep the 0% baseline as the explicit fallback.

Advance at most one candidate. It must improve full-label RMSE on **both** assignments and have mean relative improvement **>=0.20%**, approximately INR 39,300 at this local baseline. Rank qualifying candidates by mean relative gain, then lower new-model weight, fewer components, then route ID. Lock the entire recipe and weight before confirmation. An engineering failure is not an accuracy loss; a local gain is not a portal gain.

Report standalone errors, residual correlations with V14, per-fold gains, <=40M / 40–80M / >80M contributions, missing/unseen-city results and sensitivity excluding V14's five largest errors. These diagnostics explain results; they do not delete labels or create new routes. In particular, residual correlation alone is not the selection criterion.

## Confirmation and final delivery

Predeclare three additional five-fold random assignments **919601, 919602 and 919603**, plus inherited city-group and target-stratified checks. Confirm only the locked candidate. Rebuild matched V14 controls for the new random assignments: fifteen original-platform V7, fifteen T9 and fifteen P35_8 fits. Reuse city/stratified controls only with verified provenance. No full-data fitted model may predict its own validation rows.

The candidate requires 25 confirmation folds per selected component: at most **75 new component outputs** if the three-core mean wins. The new random V14 controls add **45 component fits**. Include these costs before development admission, rather than treating confirmation as a cheap afterthought.

Keep the existing confirmation policy: positive gains on all three new random assignments, mean relative random gain >=0.20%, positive lower 95% paired-row gain bound from 5,000 resamples (seed 8319, same restaurant draws across assignments), and <=1% deterioration on each city/stratified check. No second finalist or retuning after confirmation failure. These are new partitions of historically examined labels, **not an untouched holdout**; the interval is conditional robustness evidence and does not remove search or hidden-test uncertainty.

Only after qualification, fit the full declared recipe and run the standalone notebook again in a fresh runtime/process environment. Verify 500 rows, original test ID order, finite nonnegative INR predictions, notebook/CSV agreement under the declared numerical rule, and package/source/weight hashes. Cached expected predictions may verify a computation, never replace the notebook's fitted model. Deliver the notebook, CSV, locks, measured report and reproducibility records together.

At most one qualified manual portal trial is useful for this plan. Record the actual returned score against the exact CSV hash. Replace V14 only if the observed portal RMSE is lower. A one-time coding upload depends on the actual competition state, which was not checked during this planning task.

## Proposed resource envelope and current status

Propose **eight elapsed hours** for the execution campaign, including implementation, dependencies, downloads, failures, interruptions and recovery, with **1.5 hours reserved for fresh notebook verification and delivery**. The remaining 6.5 hours form a shared research pool; the model count shrinks before scoring if the measured full-workload forecast cannot fit. Stop early when declared experiments finish. The closed V14–V18 ledgers remain unchanged; this planning task has not started a V19 execution clock.

Colab free GPU type, availability and session duration are variable. Checkpoint complete folds locally, resume only exact matching inputs/configurations/sources, and release the owned runtime at completion. An eight-hour campaign is a ceiling, not a promise of eight uninterrupted GPU hours. [Official Colab FAQ](https://research.google.com/colaboratory/faq.html).

The replacement competition deadline is not recorded in the supplied material. Do not reuse an obsolete deadline. At execution, shorten the cap to leave at least one hour before the actual deadline and preserve the delivery reserve; if a complete campaign cannot fit, retain V14.

Current state: **new account redacted-email verified; zero active assignments at the check; training not started; no V19 score or submission exists.** See [account_check.json](account_check.json). The scored V14 [CSV](reference/scored_v14.csv) and [executed notebook](reference/scored_v14.ipynb) are preserved byte-for-byte.
