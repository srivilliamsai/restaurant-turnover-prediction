# V21: two feature recipes on the successful V14 backbone

Authorized by the user's October 29, 2026 reply "start". Preserve the exact scored V14 CSV and matching notebook: observed portal RMSE 11,736,110.7877709. Free T4 only; no automatic portal upload. Six-hour operational ceiling includes implementation, setup, interruptions and recovery; reserve 1.5 hours for delivery. Previous version ledgers are unchanged.

## Frozen hypothesis and settings

V10 tested structured and group-context representations with TabPFN-v2. Their value with V14's successful TabPFN-3.5 backbone is unproven. Test exactly S (56 columns) and C+S+G (66 columns), reusing V10's reviewed definitions and literal city map. Keep the exact V14 checkpoint, eight estimators, seed 10242, float32, predictive mean, original training labels in INR millions, nonnegative output floor and fixed 500-row inference adapter. Keep checkpoint temperature/calibration defaults. No additional backbone, representation, target treatment or hyperparameter search.

S appends 16 fixed features: normalized quality mean/std/core sum, facility count, event count/mean/max, experience mean/std, social mean/product/absolute difference, age at fixed 2026-01-01, age-hygiene, age-social, and restaurant-tier times city-tier. Preserve missing operands and all-missing means.

C+S+G additionally uses V10's canonical-city view, zero-Zomato indicator and the hypothesis that zero Zomato is missing, plus eight group-context features. Fit city/cuisine counts and social/hygiene/age group means only on the current training subset; shrink means with strength 20 and use training-global means for unseen groups. Freeze group maps, categorical vocabularies and schema for every query. The Zomato interpretation is an experiment, not an established correction.

## Exactly six development comparisons

V14 = .25*original_V9 + .75*original_P35_8.

Candidate = .25*original_V9 + .75*((1-w)*original_P35_8 + w*new_P35_features), w = .25, .50, 1.00.

Two settings, ten outer folds each on assignments 42 and 71: at most 20 new development fold outputs and six selectable blends. IDs and turnover never enter predictors. Retain all 3,493 labels and evaluate ordinary RMSE in INR.

Verify input/scored-artifact hashes and all 70 inherited component-fold witnesses before GPU accuracy testing. Engineering admission uses actual fold and full-data sizes, target/ID isolation, missing/unseen inputs, stable order/batches/individual predictions, fresh fit and measured memory/runtime. Require same-layout repetition below INR .01, query-behavior differences at most INR 10 and pilot RMSE drift at most INR 1. Do not relax thresholds after results. Freeze computational sources before development.

Admit settings only from engineering evidence and a complete workload forecast with 30% contingency. If either fails, finish only the passing setting; do not add a replacement. Implementation/engineering have a 1.5-hour ceiling inside the six-hour ledger. Reserve delivery time.

Advance at most one setting/weight: positive gain on both assignments and mean relative gain >=0.20%. Rank by descending mean relative gain, smaller effective new share, then setting name. Freeze before confirmation. If none qualifies, stop and retain V14 without lowering the threshold.

## Confirmation and delivery

Use inherited matched assignments 48211/48212/48213, stratified 27421 and canonical-city groups. Verify all control provenance before reuse. These labels and partitions were examined previously; confirmation/bootstrap are conditional robustness evidence, not an untouched test.

Require all three random gains positive, mean random relative gain >=0.20%, positive lower 95% paired-row gain bound from 5,000 resamples at seed 8319, and city/stratified deterioration <=1%. One finalist needs 25 new confirmation fold fits. No retuning or second finalist after failure.

Report every comparison, fold/turnover-segment error, residual correlation and largest-error sensitivity. Diagnostics never remove labels. Independently reconstruct scores, selection and confirmation from hashed predictions.

Only a fully qualified candidate receives fresh full-data inference, a standalone freshly executed notebook and a verified 500-row CSV in original ID order. Preserve the original-Mac V7 fitted-state boundary; identical fresh Linux V7 training is not claimed. Expected predictions are witnesses after inference. At most one manual portal trial after checking portal availability; retain V14 until a lower actual portal score is reported. The previous user's "no limit" response concerned the portal deadline and does not imply paid compute or unlimited execution.

Collect evidence, release only this experiment's runtime, verify account release and package the measured report and fallback regardless of result. End this campaign if neither feature recipe qualifies.
