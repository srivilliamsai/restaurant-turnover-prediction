# Implemented V10 experiment contract

Baseline: scored V9, portal RMSE **11,904,515.9864996**, rank 9 in the supplied October 27 screenshot. Preserve its scored CSV and matching executed notebook. This six-setting contract replaces the earlier four-setting review draft.

## Fixed settings and features

All settings use official TabPFN v2 regression, raw target in millions of INR, 16 estimators, seed 10242, predictive means, and V9's pinned T4/CUDA/float32 configuration. Model source and exact checkpoint/dependency hashes are frozen before screening.

Numerical preflight amendment: native float32 inference differed by INR 1.295133 when prediction batches were split. Use fixed 500-row internal batches, padding with a frozen training-feature row and discarding padding outputs. This is an inference consistency fix, not prediction rounding or another parameter setting. Refit V9 TabPFN controls on exactly the same validation folds using the adapter, retain their original verified OOF witnesses, and report numerical differences. The original 500-row submission shape is unchanged and scored V9 must still reproduce within INR 0.01. The failed preflight and its compute time are preserved.

| Setting | Representation | Features |
|---|---|---:|
| C | V7 plus cleanup | 42 |
| S | V7 plus structured block | 56 |
| C+S | Both | 58 |
| C+S+G | Both plus training-subset group context | 66 |
| P | V7 excluding Food Rating and Order Wait Time | 38 |
| C+S+P | Both blocks, excluding those inputs before engineering | 56 |

C retains the original City representation and adds the conservative literal canonical-city view from the review manifest. It changes Zomato zero to missing in the new component and adds a zero indicator. It uses no fuzzy/substring city matching or metropolitan pooling.

S adds exactly 16 columns: mean/std of quality ratings normalized by their documented maximums (staff/8, value/7, hygiene/10, food/10, Zomato/5, overall/10); normalized hygiene+staff+value sum; five-facility count; event-offering count; mean/max normalized offered-event ratings (music/10, comedy/6, deals/7, sports/6); mean/std of the five experience ratings; social mean/product/absolute gap; age relative to 2026-01-01 bounded below by zero; age×normalized hygiene; age×social mean; tier×city tier. Means use available values and all-missing means stay missing. Standard deviations use ddof=1. Products preserve missing operands. Facility sums use min_count=1. Missing dates and tiers remain missing.

P removes both specified raw inputs before every feature builder. C+S+P recomputes quality aggregates without Food Rating; neither removed field may influence any feature or interaction.

P-input invariance applies to the new P TabPFN component. The frozen V7 and original T9 shares retain their original feature definitions under the prescribed blend; the complete blend is not claimed to ignore those inputs globally.

G adds log1p city/cuisine counts plus deviations from each group's social, normalized-hygiene and age means. Fit all maps exclusively inside each training subset. Shrink means toward its global mean with strength 20. Unseen groups have count zero and global-mean fallback. Freeze maps for all prediction batches.

## Selection

For each setting compare weights w=0.25,0.50,0.75,1.00 in:

`0.5*V7 + 0.5*((1-w)*original_V9_TabPFN + w*new_TabPFN)`.

There are six settings and 24 selectable blends. Component-alone scores are diagnostic. No parameter/loss search, clipping/filtering of targets, calibration, rounding or additional family is permitted.

Development uses the identical five-fold assignments 42 and 71. Advance at most one blend that improves full-target RMSE on both. Rank by mean RMSE, smaller replacement weight, fewer features and setting name; freeze the complete choice before confirmation.

Confirmation uses random assignments 48211,48212,48213, existing strat:27421, and revised canonical-city GroupKFold. Require improvement on all three random assignments and at most 1% deterioration on each stratified/city check. If any gate fails or no setting advances, retain scored V9. No retuning on confirmation results. Revised city membership requires fresh comparisons on exactly those folds, including original-platform V7 fitting.

Report every configuration, fold errors, turnover groups <=40M/40-80M/>80M, ten largest positive row contributions and tail-group share of net squared-error reduction. Use 5,000 paired-row bootstrap resamples across the three random assignments, seed 8319; intervals are conditional and exclude model selection/retraining/distribution uncertainty.

## Verification and delivery

Stages: audit, pilot, screen, confirm, export. Explicit data/output directories; default data is the workspace. Hash-check resumable caches and reject stale, corrupt and incomplete pairs. Include input/source/feature-map/configuration/dependency/checkpoint/training-row/fold/prediction identity.

Required checks: outer-validation label isolation; target/ID exclusion; P-input invariance; shared category meanings; missing/unseen/unknown-city/all-missing-event inputs; row permutation and separate-batch inference; exactly 500 rows with original unique IDs/order, exact required headers, finite nonnegative values; separate V9 reproduction and fresh standalone notebook reproduction within INR 0.01.

Preserve V9's reproduction boundary: its V7 component uses the user's own independently fresh Mac-fitted, hash-verified state. TabPFN fits afresh in the standalone notebook. An identical fresh Linux V7 refit is not claimed.

Deliver measured results and all failed comparisons, complete inspected-source inventory, manifests, and a qualified CSV with matching freshly executed notebook only after full qualification. No qualifying candidate means no new recommended portal upload. A qualified finalist may receive one portal trial after deadline timezone verification. Record exact CSV hash and returned score; retain the lowest observed score. Keep the one-time coding upload pending the final choice.

## Resources

Use free T4 compute only; no paid resources. Pilot at actual outer-fold size before screening. The human explicitly amended the original 18-hour total cap to **24 hours** after runtime interruptions. The six settings, 24 blends and qualification rules are unchanged. The original stage allowances remain audit 1, pilot 1, screen 6, confirm 7, delivery 3; additional total allowance covers the conservatively charged interruption intervals. `budget_amendment.json` preserves the authorization, prior ledger hash and every earlier charge. The continuation tracks only the remaining authorized epoch; add its `prior_charged_seconds` to its `seconds` for the global total. Save each completed fold persistently, reserve delivery time, record unfinished work explicitly and never export an incompletely confirmed candidate. Aim for October 28 artifacts; verify the October 29 deadline timezone before a portal upload. No first-place score is promised.
