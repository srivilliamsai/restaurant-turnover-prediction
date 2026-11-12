# Proposed V11: test two new backbones against scored V9

Accepted October 27, 2026 by the user instruction: start do faster. Execution uses the CLI, free compute only and a separate eight-hour maximum. See transport_run/status.json for live status.

Keep the scored V9 CSV and its matching executed notebook unchanged. Its observed portal RMSE is **11,904,515.9864996**. The latest supplied screenshot shows rank **11**. The recorded V7 screenshot showed rank 24 and the V9 submission screenshot showed rank 9; rank 28 is not independently documented in the reviewed artifacts. Rankings are relative standings, so use RMSE for comparisons between our files.

## What actually improved V9

V9 is **50% scored V7 plus 50% TabPFN v2**, using raw targets in millions of INR, V7's 40 features, 16 estimators and seed 10242. It retained all 3,493 original training targets.

| Evidence | V9 versus V7 | V10 versus V9 |
|---|---:|---:|
| Mean gain on three random confirmation assignments | 94,989.07 | 4,083.91 |
| Conditional paired-row 95% gain interval, INR | [35,752.50, 159,172.10] | [-6,546.86, 15,187.14] |
| Observed portal RMSE reduction; negative means worse | 119,800.55 | -21,220.51 |

V9's observed portal reduction was 0.9963%. Its stronger local evidence came from introducing a different pretrained model into the ensemble. V10's cleanup and feature experiments produced only small gains and did not improve the portal result. Passing the earlier positive-gain gates did not establish a useful hidden-test improvement.

I recomputed the three original V9 comparisons from hash-verified prediction snapshots. The results are in V9_V10_REVIEW.json. They reproduce the earlier report. The 48 restaurants above INR 80M still account for approximately 68% of squared error. This is a diagnostic, not evidence that their labels are wrong and not a reason to remove them. V9 improved overall despite worsening that segment on its three random checks; a mandatory tail-only improvement rule would therefore be unsupported.

## Resources checked and priorities

- **TabICLv2 regression:** the official implementation provides TabICLRegressor and tabicl-regressor-v2-20260212.ckpt. It uses a different architecture/pretraining procedure from TabPFN and is the first priority for new model diversity. Its authors describe synthetic pretraining. This is a hypothesis for our data, not a measured improvement. [Official repository](https://github.com/soda-inria/tabicl)
- **TabPFN-3.5 Base:** official local model, supporting regression from the shared tabpfn-v3.5-20260909.safetensors checkpoint. Its model card describes synthetic-only pretraining. It is newer than V9's explicitly pinned v2 checkpoint; its value here must be tested. Use the local package. [Official models documentation](https://docs.priorlabs.ai/models), [official model card](https://huggingface.co/Prior-Labs/tabpfn_3_5)
- **Predeclared engineering fallback:** if TabPFN-3.5 Base cannot be accessed or cannot pass the T4 feasibility gate, substitute the synthetic-only default TabPFN-2.5 regression checkpoint, tabpfn-v2.5-regressor-v2.5_default.ckpt. Decide this before development scores are examined. Do not try both backbones and then choose using development results. [Official checkpoint descriptions](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- **RealMLP and TabM:** additional neural regression options available through PyTabKit. Defer them to avoid expanding this round into another broad search. [Maintainer repository](https://github.com/dholzmueller/pytabkit)

Ridge/HGB in V8 and EBM in V9 did not qualify. Feature blocks and removal experiments in V10 did not improve the observed portal best. Turnover-band/tier models were already examined in V7. Do not repeat those grids, introduce another seven-model ensemble, or adopt a public repository's claimed score without an independently reproducible method. The earlier article's 2.893M outcome was not reproduced in our review.

## Four fixed settings

Use isolated improvement_v11 code, environments, outputs and checkpoints. Do not modify V9/V10 environments or their model sources.

| Setting | Backbone | Estimators | Target |
|---|---|---:|---|
| I8 | Official TabICLv2 regression | 8 | Raw INR millions |
| I16 | Same checkpoint | 16 | Raw INR millions |
| N8 | Official TabPFN-3.5 Base, or the single predeclared fallback | 8 | Raw INR millions |
| N16 | Same checkpoint as N8 | 16 | Raw INR millions |

Freeze seed 10242, predictive-mean output and float32 inference. TabICL uses its documented default normalization/feature-view policy; freeze the complete resolved parameters before screening. Use a fixed ensemble batch size of one where supported to bound T4 memory, without searching it for score improvements. Disable automatic mixed precision and hardware-specific attention choices where supported for the fixed reproducibility contract. Record the actual supported arguments during the pilot; do not invent unsupported API flags. If a backbone cannot meet the contract, skip it rather than silently relaxing the numerical checks.

Use precisely V9's frozen feature builder: no V10 city remapping, Zomato-zero reinterpretation, structured aggregates or input removals. IDs and targets cannot enter predictors. Category dictionaries, library preprocessing and missing-value handling must be fitted within the current training subset; unknown categories map to the documented missing/unknown representation. Preserve all original labels in training and evaluation. No target transformations, upper prediction caps, target deletion, target winsorization, fitted correction, external enrichment or global calibrator. Apply only a predetermined nonnegative output floor if necessary.

For a new model M and candidate weight w:

`prediction(w) = (1 - w) * scored_V9_model + w * M`

Test w in **0.25, 0.50, 0.75, 1.00**. This is **four settings and sixteen selectable blends**. Weight 1.00 also evaluates the model alone. Keep unchanged V9 as the control. Do not use leaderboard feedback to tune weights.

## Pilot before the full screen

Use the existing authorized Colab CLI account and a free T4. No paid GPU, paid hosted inference or larger machine is assumed. Keep credentials in the CLI store, outside notebooks and archives.

Within a maximum one-hour combined setup/feasibility allowance:

1. Preserve and hash the four official inputs and scored V9 artifacts. Verify the separate original-Mac V7 reproduction boundary.
2. Freeze exact package versions, repository revisions, checkpoint SHA256 values and complete model parameters. Install in the new environment and restart the notebook kernel after dependency changes before importing numerical packages.
3. Test each backbone at the actual outer-fold size, approximately 2,794 training and 699 validation rows. Measure fitting/prediction time and peak memory. Repeat-fit, split-batch, row-permutation, individual-row and missing/unseen-category checks must satisfy the existing INR 0.01 tolerance.
4. The fixed 500-row prediction adapter may be reused only after it passes these tests for the new backbone. Report and resolve padding/batch dependence before screening; never round predictions to force a pass.
5. Choose the TabPFN backbone solely from access, memory, reproducibility and runtime evidence. Freeze that choice before development results. If only one family is feasible, screen its two settings; do not add substitute search families.
6. Forecast full work from measured times. If the declared screen, confirmation and delivery cannot fit the budget, reduce to the two feasible settings or stop and retain V9. Never skip required gates to export a candidate.

## Development and a stronger promotion bar

Evaluate every feasible setting on identical five-fold assignments **42 and 71**. This is at most 40 new candidate fits. Reuse baseline controls only after checking input/source/configuration/platform/fold/row-order/prediction hashes and the fixed inference adapter. A cross-round cache requires explicit provenance linking its generating implementation; a matching filename is insufficient.

Define relative gain per assignment as `(V9_RMSE - candidate_RMSE) / V9_RMSE`. Require:

- Lower full-target RMSE on both development assignments.
- Mean relative gain of **at least 0.20%**, approximately INR 40,000 at the observed local baseline. This is a proposed practical decision threshold, not a guarantee or a formal significance level.

Rank eligible blends by lower mean RMSE, then lower candidate weight, fewer estimators and setting name. Freeze **one finalist**, including all preprocessing, weights, checkpoint and inference settings. Do not select by restaurant subgroup scores. If nothing passes, stop before confirmation and recommend V9.

## Confirmation

Use three predeclared new random five-fold assignments **61301, 61302 and 61303**, the existing stratified assignment **27421**, and V10's frozen normalized-city groups. These seeds are newly specified in this proposal; all rows/labels remain historically examined, so this is not an untouched external test.

For the three new random controls, fit V7 on its original Mac platform, entirely inside each outer training fold, and TabPFN v2 in the pinned CUDA environment. Both control and candidate must use identical held-out rows. The existing stratified/city V9 controls may be reused only after exact provenance and fold checks. Keep old V9 witnesses and record any numerical adapter differences explicitly. Do not replace the original-platform baseline with an unmatched Linux refit.

A finalist qualifies only if:

- It improves over V9 on **all three** new random assignments.
- Its mean random relative improvement is **at least 0.20%**.
- Its **5,000-resample paired-row 95% interval has a lower gain bound above zero**. Draw the same restaurant indices across the three assignments. Report it as conditional on the fitted models and reused labels; it does not account for the entire search or hidden-test distribution.
- Full-target RMSE on city and stratified validation deteriorates by at most **1%** each.

Report every failed comparison, fold results, turnover segments <=40M/40-80M/>80M, and the largest individual contributions to squared-error changes. Examine candidate error correlation and which V9 mistakes it corrects as diagnostics; lower correlation alone is not an acceptance criterion. Do not retune on confirmation outcomes or use true turnover bands to choose an inference-time model.

## Verification, delivery and one portal trial

A qualified finalist must pass training/validation label isolation, ID exclusion, fold-local preprocessing, missing/unseen values, row permutation, separate-batch prediction, incomplete/corrupt/stale checkpoint rejection and repeat execution checks. Baseline V9 reproduction remains separate from candidate reproduction.

Export exactly 500 rows in official test order, unique original IDs and the exact headers `Registration Number`, `Annual Turnover`, with finite nonnegative predictions. A fresh standalone notebook must reproduce its CSV within INR 0.01 in the pinned environment. Preserve the original-Mac fitted-state approach and its stated portability boundary. Deliver the executed notebook, CSV, all measured comparisons, input/source/environment/checkpoint/output manifests and a complete score-to-file ledger.

Only after every gate and reproduction check passes, permit **one** user-uploaded V11 portal trial. Record the returned score against its exact CSV hash and preserve screenshot provenance. Retain the lowest observed score. A local gain never becomes an assumed portal gain. If the candidate loses, retain V9; do not use that portal result to search nearby weights or row adjustments.

Keep the one-time coding assignment upload pending until final model choice. Verify the portal deadline timezone; the supplied page shows October 29 at 11:59 PM, without a confirmed timezone.

## Proposed compute and schedule

Proposed **8-hour maximum for V11**, not a completion estimate: one hour audit/setup/pilot, two hours screening, three hours confirmation (including original-Mac control fits), and two hours reproduction/delivery. Prior V10 accounting remains closed at 16.9253 hours; this is a separate proposed round, not an extension or reset of its ledger. The user accepted this plan with the instruction start do faster; no paid resources are authorized.

Use free resources only, save every completed fold locally and persistently, and bound allocation/reconnect attempts. Run the finite coordinator independently of the interactive chat process, with a backup watcher, active stage heartbeats and cleanup of its owned runtime. Separate actual compute, elapsed wall time and conservative interruption charges. Aim to finish artifacts October 28 and preserve October 29 as a buffer. If the pilot predicts an overrun, stop or execute only the feasible declared family. No incomplete candidate may be exported.

The intended result is a materially stronger, reproducible challenge to V9. These resources offer testable methods, not a reproducible promise of beating the earlier 4.797M leader score.
