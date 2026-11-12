# Approved V8 — complementary models within 18 hours

Baseline: scored V7 Trial 1, portal RMSE 12,024,316.5355221. Preserve its CSV and matching notebook. First place is an ambition; only an actual returned portal score establishes it.

## Fixed settings

A: four native HGB settings: raw/square-root target × 7/15 leaves.
B: four HGB settings with cross-fitted statistics: raw/square-root target × smoothing 50/200, 15 leaves.
C: eight Ridge settings: raw/square-root target × alpha 10/100 × base/structured features.

HGB uses learning rate .03, 600 iterations, minimum 50 rows per leaf, L2 20, at most 255 bins, no early stopping and seed 10242. Learn category vocabularies inside each training subset; pool counts below ten, keep at most 253 named levels plus a rare level, and map unseen levels to missing. Preserve numeric missing values.

TargetEncoder uses the continuous original target in millions and Cuisine, Restaurant Theme and Opening Year. Its fit_transform uses five shuffled inner folds, seed 831. Validation/test only use transform.

Ridge uses median imputation with missingness indicators, standardized numeric values, one-hot minimum frequency ten, unknown-category handling, an intercept and Cholesky solving. Structured Ridge adds exactly: social mean, age relative to 2026-01-01 bounded below by zero, social×tier, social×hygiene, age×social, tier×city-tier, log1p(training city count), log1p(training cuisine count). Unknown counts are zero; compute interactions before imputation.

Fit targets in millions. Truncate square-root predictions at zero before squaring and converting back to INR. No inverse correction, target filtering, external data, new tail model or global calibrator.

Each component alone and weights .25/.5/.75 with V7 gives 64 development configurations.

## Promotion gates

1. Same five-fold development assignments 42 and 71 for candidates and hash-verified V7.
2. Advance only improvements on both assignments: at most one HGB approach from A/B and one Ridge from C. Rank by mean RMSE, lower candidate weight, then name.
3. For advancing settings, test five-fold model averaging (inner seed 831) and a 50:50 mix with the full-subset fit. Refit preprocessing inside every inner subset. Maximum 20 settings / 80 configurations.
4. Freeze before confirmation on random assignments 48211/48212/48213, target-stratified 27421 and normalized-city groups.
5. Require positive gain on all three random confirmations, and no more than 1% deterioration on city or stratified validation. Otherwise retain V7.
6. Report all original-target RMSE, bins <=40M / 40–80M / >80M and fold consistency. A 5,000-resample paired-row interval across the three random assignments is calculated only for actual confirmation results. All labels were examined historically.

## Execution and delivery

Isolate audit, screen, refine, confirm and export stages under improvement_v8. Accept explicit data/output directories, default to the workspace's four official CSVs, and pin working package versions. Resume only when inputs, source, configuration, runtime, training rows/targets, validation rows and prediction hashes match.

Maximum 18 hours: audit 1, screening 5, refinement 2, confirmation 7, reproduction/delivery 3. Finish earlier when the declared experiments end. Local CPU first; optional Colab and persistent checkpoints; no paid resources assumed. Do not expand the fixed search after observing failures.

Verify own-label exclusion, outer-validation label isolation, training-only counts, stale/corrupt/interrupted caches, missing/unseen values, permutation, IDs/target exclusion, 500 original restaurant IDs/order, exact two columns and finite nonnegative predictions. Fresh CSV reproduction and scored-V7 reproduction must agree within INR .01.

Deliver measured reports, source/provenance and hash manifests, an executed standalone research notebook and matching notebooks/CSVs for any qualified candidates. If none qualify, recommend byte-identical scored V7 files. Record only actual portal scores against exact CSV hashes; retain the lowest observed score. Finish by October 28 where possible, confirm the deadline timezone before uploading and choose the one-time coding notebook only after final model selection.

The author repository was reviewed before training. Public code/predictions, copied weights, target filters and leaderboard-derived row adjustments are excluded. Its reported low leaderboard stage is not treated as a reproducible training recipe.
