# Accepted V12: strengthen V9 and test one complementary model

Authorized October 28, 2026: PLEASE IMPLEMENT THIS PLAN. Original twelve-hour cap extended by the user's explicit reply: "Raise to 18 hours; complete all four settings (Recommended)" and instruction "dont skipp check all correctly running". Maximum eighteen hours with all earlier V12 charges retained, free compute through Colab CLI, original V7 control fits on Mac. Preserve scored V9 RMSE 11,904,515.9864996, CSV and executed notebook. No automatic portal upload or paid fallback.

Four fixed settings, raw targets in INR millions, all 3,493 labels, V9's forty source predictors, no IDs/targets as inputs:

| Name | Definition |
|---|---|
| T32 | Official V9 TabPFN-v2 checkpoint, 32 estimators, seed 10242 |
| T4x16 | Equal mean of 16-estimator models, seeds 10242/20242/30242/40242 |
| R1 | RealMLP-TD, seed 10242 |
| R4 | Equal mean of RealMLP-TD seeds 10242/20242/30242/40242 |

TabPFN package 9.2.0, exact official v2 checkpoint/revision/hash from reference_manifest.json, predictive mean, low-memory, float32, T4. Family T uses `.5*V7+.5*((1-w)*T9+w*Tnew)`; family R uses `(1-w)*V9+w*R`. Only weights .25/.50/.75/1: maximum sixteen selectable configurations. Constituent models share verified fold checkpoints. Final predictions have a nonnegative floor, no upper clipping or rounding.

RealMLP uses pytabkit 1.7.3, hidden layers [256,256,256], at most 256 epochs, learning rate .2, weight decay .02, dropout .15, MSE loss and inner RMSE selection. Fixed 20% inner validation indices seed 831, n_cv=1, n_refit=1 on all outer training rows. Library output normalization on, clamping off. Numeric zero fill and a missing indicator for every numeric source column; training-subset categorical vocabulary with reserved unknown category. Other resolved release defaults, dependencies and seeds are frozen before screening. No new feature/loss/checkpoint search or target filtering.

Stages: audit, pilot, screen, confirm, export. Pilot at real outer-fold size tests repeated fits, row/batch/individual invariance, missing/unseen inputs, time and memory. All four settings must finish their checks; an interrupted check must be resumed rather than silently omitted. All sixteen blends must finish development before selection. Genuine engineering failures or insufficient total budget are reported and block advancement; they never waive verification. Feasibility and budget decisions use no development RMSE.

Development: identical five-fold assignments 42 and 71, original-scale full RMSE. Require positive gains both and mean relative gain >=.002. Choose at most one per family; rank mean RMSE, lower weight, lower inference cost, name. T inference cost is total TabPFN passes; R cost is number of networks. Family membership means those scales never break a within-family tie across architectures. Freeze trial order by development rank.

Confirmation: random five-fold assignments 71501,71502,71503; stratified 27421 and frozen V11 normalized-city groups. Reuse only fully provenance-verified controls; new random V7 fits on original Mac and TabPFN control on pinned T4. Require positive gains on all three random assignments, mean random relative gain >=.002, 5,000 paired-row seed-8319 95% lower gain bound >0, city/stratified deterioration <=1%. No retuning on confirmation. Historically examined labels and conditional intervals are robustness evidence, not an untouched test.

Report every fold, failed comparison, error correlation, <=40M/40-80M/>80M errors and concentration of row-level squared-error gains. These groups cannot choose inference-time models.

Updated eighteen-hour budget guards: audit/setup/pilot at most 4h, development 3h, confirmation including Mac at most 8h, reproduction/delivery reserve 3h. Shared fifteen-hour pre-delivery and eighteen-hour total limits take precedence; earlier charges are never reset. Working IST deadlines October 29: training 18:00, artifacts 21:00, portal trials 22:00, competition 23:59. Verify portal timezone before user submission. Any unfinished work is explicit, no incomplete candidate export.

Before GPU allocation smoke-test actual entrypoint, resume/corrupt cache and no-finalist branches. Verify outer-label/ID isolation, category alignment, missing/unseen values, row permutations and prediction batches, repeat fitting and fresh notebook reproduction within INR .01. CSV exactly 500 unique original IDs/order and headers Registration Number, Annual Turnover; finite nonnegative predictions. Verify V9 separately, retaining original-Mac fitted-state V7 boundary.

Only qualified finalists get CSVs and matching freshly executed standalone Colab notebooks. At most two user portal trials, one per independently qualified family, frozen order, actual scores recorded against exact hashes. Retain lowest observed score. No nearby-weight tuning from portal feedback. Keep one-time coding upload pending final choice. If nothing qualifies, preserve original V9 files and recommend no new upload.

Sources: https://github.com/PriorLabs/TabPFN ; https://github.com/dholzmueller/pytabkit ; https://pypi.org/project/pytabkit/1.7.3/ . General-purpose official TabPFN checkpoint only; no competition-specific external labels or competitor predictions. No first-place result promised.
