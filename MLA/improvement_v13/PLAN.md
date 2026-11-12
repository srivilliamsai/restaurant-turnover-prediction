# V13 fixed execution contract

Approved maximum: eight hours, free compute only. Baseline: scored V9, portal RMSE 11,904,515.9864996.

Use the unmodified forty-column V9 features, official pinned TabPFN-v2 regression checkpoint, package 9.2.0, sixteen estimators and seed 10242.

Three settings: Q99/Q995 cap training labels at the current training subset's 99th/99.5th percentile using linear interpolation. SQ2 trains on sqrt(y/1e6), then predicts criterion.mean_of_square(logits)*1e6. Save squared-mean predictions for diagnostics. Retain all training rows and every original validation label. Extreme labels are not established errors.

Evaluate (1-w)*V9+w*M at w=25/50/75/100% on five-fold development assignments 42 and 71. Advance one candidate with positive gains on both and mean relative gain >=0.20%. Rank by descending mean relative gain, lower candidate weight, then experiment name. Freeze before confirmation on 48211/48212/48213, stratified 27421 and the inherited canonical-city grouped assignment.

Confirmation requires positive gains on all three random assignments, mean relative gain >=0.20%, positive lower 95% paired-row gain bound (5,000 draws, seed 8319), and city/stratified deterioration <=1%. Previously examined labels provide conditional robustness evidence, not an untouched test. Report all configurations, folds, component scores, <=40M/40-80M/>80M segments and concentration diagnostics. No finalist means retain the byte-identical scored V9 files.

Stage allowances: audit 0.5h; pilot 1h; development 2h; confirmation 2.5h; delivery 2h. Preserve two hours for delivery. Persist global elapsed time, interrupted work and complete fold checkpoints. Admit settings by actual-fold feasibility before inspecting development scores. If the measured remaining workload cannot fit, stop and retain V9.

Reuse controls only after input/source/runtime/row/fold/target and output hashes are verified. Use the fixed 500-row adapter, padded with the first encoded training row. Training percentile thresholds and treatment identity are part of each cache key. IDs and targets cannot enter features.

A qualified CSV requires a separate freshly executed standalone notebook reproducing it within INR 0.01. Preserve 500 original test IDs/order and exact Registration Number,Annual Turnover headers. Check target isolation, moment integration, missing/unseen inputs, permutation, batching, repeated fits and malformed outputs. V7 uses the existing original-Mac independently verified fitted state; both TabPFN components fit afresh. No identical Linux V7 retraining claim.

After qualification, reconfirm portal deadline/timezone, permit one manual trial, record actual score against the exact CSV hash, and retain whichever observed submission is best. Keep the one-time coding upload pending. No automatic portal submission or promised rank.
