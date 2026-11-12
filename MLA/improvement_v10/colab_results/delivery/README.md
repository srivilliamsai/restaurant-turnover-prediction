# V10: six fixed feature experiments

Current observed portal best remains **V9: RMSE 11,904,515.9864996**. Its scored files are preserved in `reference/` and the earlier V9 directory.

The workflow tests C, S, C+S, C+S+G, P and C+S+P using fixed 16-estimator raw-target TabPFN v2. Each tests replacement weights 25%, 50%, 75% and 100% within V9's TabPFN half; V7 retains 50% of every final blend. There are 24 selectable configurations. No target filtering, rounding, additional losses or parameter searches are included.

Use `python workflow.py STAGE --data-dir DATA --output-dir OUTPUT`, where STAGE is `audit`, `pilot`, `screen`, `confirm`, `export` or `all`. Default data directory is the workspace, and default output is `improvement_v10/results`. Maximum accumulated stage budget is 18 hours, including a three-hour export/reproduction reserve. Every completed model fold is saved atomically with source/configuration/input/fold and prediction hashes. Changed, corrupt and incomplete checkpoints are rejected.

**Current continuation amendment:** the human raised the total cap to 24 hours, free compute only. `results/budget_amendment.json` preserves the prior 12.289-hour conservative charge and its ledger hash. The continuation's `compute_budget.json` counts a new remaining epoch; add `prior_charged_seconds` to `seconds` for the global total. Its `maximum_hours` is the remaining epoch allowance, not a new 24-hour allocation. Pass that value as `--budget-hours` when manually resuming a stage. Reuse the hash-verified passing audit/pilot and resume `screen`, rather than rerunning `all` against the original frozen audit's 18-hour budget field. Model sources, features, settings, folds and acceptance gates remain fixed.

The four official input filenames are Train_dataset.csv, Test_dataset.csv, Data_Dictionary.csv and Sample_Solution.csv. Their hashes must match the frozen official data. Restaurant IDs and every original target remain unchanged; IDs never enter predictors. Category vocabularies and group maps are learned on the current training fold. The P settings remove Food Rating and Order Wait Time before all feature engineering.

Development uses five-fold assignments 42 and 71. Advance at most one blend that improves full-target RMSE on both, ranked by mean RMSE, lower replacement weight, fewer features and setting name. Freeze before confirmation on 48211, 48212, 48213, strat:27421 and normalized-city groups. Require improvement on all three random assignments and at most 1% deterioration on stratified/city checks. Otherwise retain V9. Report every failed configuration and target-group diagnostics.

The revised city map changes geographic group assignments. If a finalist advances, `refit_city.py` fits its V7 comparison on the original Mac platform using the exact revised folds. The GPU workflow waits for that verified artifact rather than silently using old city predictions or a numerically different Linux V7 refit.

The final notebook loads the user's own independently verified V7 fitted state and trains the required TabPFN components afresh. Fresh Linux V7 retraining previously failed reproduction and is not claimed to reproduce the scored baseline. A separate fresh notebook process must reproduce its CSV within INR 0.01. Expected CSV values are used only for verification after fitting/prediction.

The first numerical pilot detected INR 1.295133 dependence on prediction batch size. The inference adapter now uses fixed 500-row batches with a frozen training-feature padding row; padding outputs are discarded. No prediction rounding is used. Validation TabPFN controls are refitted with this same adapter and compared with preserved original OOF witnesses. The full 500-row scored submission shape is unchanged. The original failure is preserved, and its compute time remains charged.

Only the authorized free T4 is used. No paid compute or portal upload occurs automatically in the training workflow. A qualified CSV may receive one portal trial after confirming the deadline timezone; retain the lowest observed portal score and keep the one-time coding upload pending the final choice.

The official Colab CLI now executes real training in the foreground of a notebook kernel. Earlier detached workers lost their free runtime; those interrupted diagnostics and elapsed-time charges are preserved. Completed fold pairs are backed up locally and checked before reuse. Local live logs can be viewed with `tail -f improvement_v10/colab_results/run.log` from the workspace. A running status alone is not a passing pilot or a measured improvement.

The finite continuation driver and local backup watcher run in an OS process group outside the interactive chat tool session. Their PID, phase and logs are in `transport_run/`. The driver enforces the remaining authorized time, collects completed stages, performs the original-Mac city comparison after finalist freezing, and releases only its owned free T4 at completion/failure. No scheduled or recurring job is installed.

The Gemini export inventory is provenance only. None of those scripts is executed. Their global target-encoding leakage, restricted-target blend optimization, independent category coding and incomplete drafts are not adopted.

TabPFN attribution: Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. The official v2 regression checkpoint is pinned to revision 4972a65a1b30806315c6f92499959ffbfc69a673 and SHA256 2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736. No competitor predictions or externally supplied competition-trained weights are used. The preserved V7 state was trained in your own earlier workflow on the official data.
