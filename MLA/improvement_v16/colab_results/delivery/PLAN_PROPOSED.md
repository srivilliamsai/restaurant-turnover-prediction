# Proposed V16: test a different regression architecture

Prepared October 29, 2026 (IST). Status: research and planning only. No V16 model has been trained, no runtime has been allocated and no new submission is claimed.

**Recommendation: make TabM with numerical embeddings the primary experiment. Admit one Mitra-v2 fine-tuning recipe only if its full engineering pilot fits the same free-T4 budget. Keep scored V14 as the control and fallback.**

## 1. What V15 actually established

| Evidence | Measured result | Consequence |
|---|---|---|
| Scored V14 | Portal RMSE INR 11,736,110.7877709 | This is the current observed best. |
| V15 retry | 50 development fold fits, 20 blends; best mean gain INR 4,487.59, or 0.022838% | It missed the existing 0.20% gate. No V15 portal trial occurred. |
| V15 candidate errors versus V14 | Correlation 0.99560 to 0.99846 across the saved development comparisons | The tested changes made very similar errors; another temperature/view grid has weak support. |
| V12 | RealMLP R1/R4 failed development | Do not present RealMLP as a new untested model. |
| V14 | TabICLv2 completed screening; TabPFN-3.5 supplied the qualified improvement | Do not repeat TabICLv2 as unfinished work. |

The V15 winning blend was 25% V9 + 37.5% original V14 TabPFN-3.5 + 37.5% N32-T100. Mean development RMSE changed from INR 19,649,469.51 to 19,644,981.91. The 0.20% practical gate corresponds to roughly INR 39,299 at this local baseline; the observed gain was about one ninth of that amount.

These correlations support deprioritizing the tested refinement directions. They do not prove that TabPFN has reached its best possible accuracy or that a different architecture will win.

Primary local evidence: [V15 measured result](../improvement_v15_retry1/delivery/V15_Result_Summary.md), [all 20 comparisons](../improvement_v15_retry1/results/screening.csv), [component diagnostics](../improvement_v15_retry1/results/component_diagnostics.csv), and [V14 portal attribution](../improvement_v15_retry1/reference/v14_portal_trial.json).

The root CURRENT_BEST.md still names V9. Use the newer V14 portal record above for this experiment. Preserve the V14 CSV SHA256 `39d49f79beb8549d04ffc78d134aad2407dc5e73914dc3c30e8bb162a3c0ba05` and its matching executed notebook.

## 2. What the internet research changes

### Primary: TabM with numerical embeddings

The authors provide a shared-weight MLP ensemble with explicit regression support and numerical embeddings. Members train with separate losses; their outputs are averaged for inference. That makes it a different architectural hypothesis from V15's fixed pretrained transformer refinements. The official example supplies a usable training reference. [Official repository](https://github.com/yandex-research/tabm), [official example](https://github.com/yandex-research/tabm/blob/main/example.ipynb).

The proposed benefit is complementary errors, not an assumed standalone win. RealMLP's failure here reduces confidence in neural alternatives generally, but does not measure TabM. Workspace review found TabM deferred in V11 and no completed TabM experiment.

### Conditional second setting: actual Mitra-v2 regression

The official Mitra-v2 model card describes a 1,000-bin predictive distribution and a point prediction obtained from its mean. It requires the official regression adapter; simply giving the v2 weights to stock Mitra-v1 regression is insufficient. [Official model card](https://huggingface.co/autogluon/mitra-regressor-2).

Its published recipe is a 50-step fine-tune with eight bag children. Fitting is triggered by prediction in the documented wrapper. Published timings use much stronger hardware and do not establish T4 feasibility. A complete outer-fold pilot must measure training plus prediction before this setting enters development. [Official package and recipe](https://huggingface.co/autogluon/mitra-finetune).

Use a supported T4 attention/dtype path. Standard FlashAttention kernels targeting newer NVIDIA architectures cannot be assumed to work on T4; the project points Turing users to a separate implementation. Do not spend this deadline building a new attention backend. [Official hardware requirements](https://github.com/Dao-AILab/flash-attention/blob/main/README.md).

### Defer for this round

TabPFN fine-tuning is relevant now that its backbone has worked here, but its official regression example recommends 80 GB CUDA memory. That recommendation is not proof that smaller configurations are impossible; it is insufficient support for making fine-tuning the main free-T4 deadline plan. [Official example](https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_regressor.py).

The October 2026 regression-objective paper reports promising aggregate quantile-loss results. Defer its objective changes, input-dependent smoothing and extra tuning until a bounded TabM MSE baseline has been measured. Its benchmark results do not establish an improvement on these restaurants. [Paper and author code](https://arxiv.org/abs/2610.13909).

## 3. Freeze two settings before development

| Setting | Fixed experiment | Admission |
|---|---|---|
| TM32-PLE | TabM, 32 shared-weight members, piecewise-linear numerical embeddings, ordinary MSE | Primary, after engineering passes |
| MITRA2-FT | Official v2 regression checkpoint and official 50-step, eight-bag recipe; distribution mean | Only if full pilot and workload forecast pass |

Both use the frozen V14 forty-column feature builder and all original labels. Reuse its categorical normalization and city aliases, including the supplied train/test endorsement-header mismatch. Targets stay in raw INR millions; no cap, log/square-root transform, target removal or target-dependent inference routing is introduced.

The following TabM settings are a proposed bounded configuration, not a claim that the authors selected these settings for this dataset:

- Backbone: `arch_type=tabm`, `k=32`, three blocks, width 256, dropout 0.1, scalar regression output.
- Embeddings: training-only quantile bins, at most 32 bins per numerical feature, dimension 16, version B, no extra activation; handle constants with the library's documented construction.
- Numeric inputs: training-only median imputation, fixed missingness indicators, then the official example's normal-quantile preprocessing with `max(10, min(n_train // 30, 1000))` quantiles. Fit its tiny deterministic jitter only on the fitting subset.
- Categories: fitting-subset vocabularies, explicit missing and unseen levels, documented categorical interface. No integer ordinality for categorical features.
- Targets: affine standardization from the fitting subset only; invert it before scoring. MSE therefore retains the original regression objective.
- Optimizer: AdamW, learning rate 0.002, weight decay 0.0003; batch size 256; gradient norm limit 1; float32, no AMP or compilation; seed 10242.
- Epoch selection: fixed 15% inner validation, seed 9091 + outer-fold index, maximum 256 epochs, patience 20. Select by unrounded inner original-scale RMSE, then start a fresh model and fit all outer-training rows for that many epochs. Refit preprocessing on those outer-training rows. The outer validation labels never choose epochs.
- Average the 32 member predictions. Train the mean of member MSEs, rather than MSE of their mean. Apply a predeclared zero floor to component output; no upper clipping.

For full-data export, repeat the same inner epoch selection within all 3,493 training rows, then refit on all rows. This is part of the frozen algorithm and is included in the runtime forecast.

For Mitra-v2, use the official regression checkpoint, seed 10242, raw-million targets, eight complete bag children and the package's frozen 50-step recipe. Keep other recipe defaults pinned to an exact package/source revision. Set external-validation-in-support off, and pass only the outer-training data to fitting. Any held-out support allowed by the recipe must be inside the outer-training data. Record actual completed steps, contexts, dtype and children. A time-truncated or fewer-child run is not silently substituted for this setting.

A Mitra `predict` call may launch another fine-tune. Persist and reuse the same fitted child models for numerical probes through a supported interface, or reject the setting. Do not call the fitting wrapper once per prediction batch or probe and mistake repeated training for ordinary inference.

## 4. Data and validation audit before accuracy screening

The read-only audit reconfirmed 3,493 training rows, 500 test rows, 48 targets above INR 80M and a target maximum of INR 400M. Before city aliases, simple whitespace/case normalization finds 24 test rows with unseen city strings and 65 with missing-city sentinels. These are input-handling checks, not evidence of hidden targets or final encoded-vocabulary counts. See [review evidence](review_evidence.json).

Use V14 predictions on exactly the same outer folds. Verify input, row, fold, source, model and output hashes before reusing controls; reconstruct V14 as 0.25*V9 + 0.75*original P35_8 where required. Do not replace the verified original-Mac V7 state with an unmatched Linux refit.

Report ordinary full-target INR RMSE as the sole primary score. Also report <=40M, 40-80M and >80M bands; missing/unseen-city rows; residual correlations; and largest squared-error contributions. Excluding the five largest baseline errors is a sensitivity diagnostic only. Retain every label in primary selection.

The earlier V14 review found that the small >80M tail dominated V9 squared error. Recompute those shares for V14; do not assume the old exact percentages still apply. The local/portal RMSE gap alone does not establish label errors or a hidden-test distribution.

## 5. Bounded development and one frozen finalist

Use the existing five-fold assignments 42 and 71. For each admitted component M, evaluate only:

```text
candidate = (1 - w) * V14 + w * M
w = 0.10, 0.25, 0.50, 0.75, 1.00
```

At most two settings, 20 new outer-fold evaluations and 10 selectable blends. There is no cross-blend between the new families, learned stacking, feature grid or nearby-weight search. Standalone components are already included at w=1.

An outer TabM evaluation includes inner epoch selection plus an all-outer-training refit: up to 40 actual training runs for development. An outer Mitra evaluation includes eight bag children: up to 80 child fine-tunes for its ten development folds. Budget model names are not equivalent to individual training runs.

Advance only a blend that improves both assignments and achieves mean relative RMSE gain >=0.20% versus V14. Rank by descending mean relative gain, then lower new-component share, then setting ID. Freeze at most one complete configuration before confirmation. If none passes, stop and retain V14.

Confirm on 48211, 48212, 48213, stratified 27421 and the inherited canonical-city grouped folds. For TabM city confirmation, its inner epoch split must also hold out complete cities inside the outer-training subset. Use analogous group-safe internal splits for Mitra if supported; otherwise disclose the inner-policy limitation and do not claim fully group-isolated tuning.

Require positive gain on all three random assignments, mean random relative gain >=0.20%, a positive lower 95% paired-row bootstrap gain bound (5,000 draws, seed 8319), and deterioration <=1% on each stratified/city check. Resample identical restaurant indices across random assignments. Report failures and do not retune from confirmation.

At most 25 finalist outer-fold evaluations are needed for confirmation: up to 50 TabM training runs or 200 Mitra child fine-tunes. Historically examined labels provide conditional robustness evidence; new split seeds do not create an untouched test.

## 6. Engineering gates and proposed compute cap

Propose a separate maximum of **six elapsed hours on free T4**, including setup, downloads, interruptions and recovery. This is a proposal, not a started run or an extension/reset of previous ledgers.

| Stage | Maximum allocation |
|---|---:|
| Audit, environment and engineering pilots | 1.0 h |
| Development | 1.5 h |
| Frozen confirmation | 1.5 h |
| Fresh execution, reporting and delivery reserve | 2.0 h |
| Total | 6.0 h |

Pilot TabM first. Give Mitra at most the remaining setup/pilot allocation, with a 30-minute ceiling of its own. Its pilot must complete the full eight-child recipe on approximately 2,794 training and 699 query rows, plus a full-training-size memory/time probe. If setup, supported deterministic inference, memory or the full workload forecast fails, exclude Mitra before reading development accuracy. Continue with TabM only. Do not replace it with another accuracy-selected family.

Forecast every remaining stage with measured fit-plus-predict times and 30% contingency. For the development admission decision, reserve for the slower possible finalist's 25 confirmation folds and full-data/fresh-notebook work. Include repeated engineering fits and Mitra prediction-triggered training. Protect the two-hour delivery reserve. If the required work does not fit, close the run; do not discard mandatory checks to create a CSV.

Require exact target/ID feature invariance, finite predictions, missing/unseen-input handling, fresh same-layout reproduction within INR 0.01, and full-fold row-permutation/split-batch plus 12 individual-row probes within INR 10, with labeled-pilot RMSE change <=INR 1. Preserve V14's fixed inference contract for its own components. Freeze each new component's supported inference layout before development; do not round predictions to pass.

Pin versions, source revisions, checkpoint hashes and resolved parameters after engineering. Failures are engineering exclusions, not accuracy failures. Preserve all elapsed charges and complete fold checkpoints across interruptions. No paid compute is proposed.

The current working schedule remains training stop by **October 29, 18:00 IST**, artifacts by **21:00 IST**, and any manual trial by **22:00 IST**. The historical portal showed October 29 at 23:59 without a confirmed timezone. Verify the real deadline before execution; the earlier real cutoff governs. These times do not establish that a six-hour run can start at any later hour.

## 7. Delivery decision

A qualified finalist requires a freshly executed standalone notebook reproducing the exported CSV within INR 0.01, without using expected predictions as inputs. Fit new components afresh. Preserve the disclosed original-Mac V7 fitted-state boundary.

Check exactly 500 original test IDs/order, exact `Registration Number,Annual Turnover` headers, finite nonnegative INR output, CSV round-trip and input/source/model hashes. Reject incomplete or stale checkpoints. Deliver all comparisons, fold predictions, qualification reconstruction, runtime ledger and either the qualified files or the byte-identical V14 fallback.

Recommend at most one manual portal trial after all gates pass. Record the actual returned score against the exact CSV hash; retain the lowest observed submission. Keep the one-time coding submission pending final model choice.

V16 is a test of different predictive behavior. It contains no promised leaderboard rank or RMSE reduction. If these tests fail, keep V14 and close this bounded search before the deadline.

