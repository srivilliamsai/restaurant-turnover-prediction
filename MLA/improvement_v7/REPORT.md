# Improvement 7 — measured research and experimental portal trials

**Student:** submission · submission · **Team:** submission

**Current decision, updated October 26:** retain **v7 Trial 1**, associated with the user's new screenshot showing **12,024,316.5355221 RMSE and rank 24**. This improves the previous **12,046,722.3691139** by **22,405.8335918 (0.185991%)**. The association with Trial 1 follows the preceding upload recommendation and conversation; the screenshot itself does not display a filename or CSV hash. Trial 2 has no reported portal score. See `portal_result_2026-10-26.json` for evidence and attribution.

**First-place feasibility:** None of these exports can mathematically reach the shown leader under the stated score assumptions. They are incremental research trials.

## Data audit and scope

- 3,493 training rows; 500 test rows. All original labels retained; **zero verified unit errors**.
- 48 targets exceed INR 80M. Only 1 of their nearest feature neighbors also exceeds INR 80M; this is a diagnostic, not proof that any label is wrong.
- No identical full-feature train/test rows. Target and restaurant IDs are excluded from features.
- Used only supplied data and original implementations. No external target matching or portal-fitted row changes.
- User-supplied rules specify RMSE across test predictions, unlimited CSV submissions and one final coding submission.

## Completed bounded research

Twelve settings: four turnover-band models (depth 3/5, with/without inner-OOF temperature calibration), four tier-specialist settings (depth 4/6, pooling strengths 100/300), and four residual corrections (Ridge alpha 100/1000; shallow CatBoost L2/Huber). Two model seeds and both five-fold development assignments 42/71 were used. Candidate weights were 25%, 50%, 75% and 100%. Ridge fits are deterministic; their repeated seed fits do not constitute independent evidence.

Best development configuration in each family (positive gain means improvement over v5):

| name | family | weight | gain_42 | gain_71 | mean_rmse |
| --- | --- | --- | --- | --- | --- |
| residual_tree_l2 | residual | 0.75 | 1,422.55 | 49,471.02 | 19,803,296.56 |
| tiers_d4_shrink300 | tiers | 0.25 | -5,613.96 | 2,492.77 | 19,830,303.94 |
| bands_d5_cal0 | bands | 0.25 | -24,014.83 | -9,789.26 | 19,845,645.39 |

All 48 model/blend configurations, including losing settings, are in `results/screening.csv`.

Residual training used three-fold v5 predictions generated entirely inside each outer training fold. These nested predictions were shared only when training rows, inputs, code, configuration and hashes matched. Tier specialists shrink toward a global model and fall back to it for small or unknown groups. Band models output probability-weighted turnover means; the final evaluation is regression RMSE.

## Frozen confirmation

| name | family | weight | scheme | v5_rmse | candidate_rmse | gain |
| --- | --- | --- | --- | --- | --- | --- |
| residual_tree_l2 | residual | 0.75 | 18421 | 19,863,662.63 | 19,832,420.14 | 31,242.49 |
| residual_tree_l2 | residual | 0.75 | 18422 | 19,816,733.50 | 19,798,032.98 | 18,700.52 |
| residual_tree_l2 | residual | 0.75 | strat:27421 | 19,828,989.58 | 19,788,419.32 | 40,570.26 |
| residual_tree_l2 | residual | 0.75 | city | 20,136,850.76 | 20,113,476.99 | 23,373.77 |
| tiers_d4_shrink300 | tiers | 0.25 | 18421 | 19,863,662.63 | 19,868,640.34 | -4,977.71 |
| tiers_d4_shrink300 | tiers | 0.25 | 18422 | 19,816,733.50 | 19,819,044.13 | -2,310.63 |
| tiers_d4_shrink300 | tiers | 0.25 | strat:27421 | 19,828,989.58 | 19,834,990.79 | -6,001.21 |
| tiers_d4_shrink300 | tiers | 0.25 | city | 20,136,850.76 | 20,147,551.18 | -10,700.42 |
| bands_d5_cal0 | bands | 0.25 | 18421 | 19,863,662.63 | 19,882,829.42 | -19,166.79 |
| bands_d5_cal0 | bands | 0.25 | 18422 | 19,816,733.50 | 19,840,755.33 | -24,021.83 |
| bands_d5_cal0 | bands | 0.25 | strat:27421 | 19,828,989.58 | 19,841,828.98 | -12,839.40 |
| bands_d5_cal0 | bands | 0.25 | city | 20,136,850.76 | 20,100,616.03 | 36,234.72 |

The best eligible configuration per family was frozen before confirmation on 18421, 18422, target-stratified five-fold seed 27421 and cleaned-city groups. v5 uses exactly the same folds. Previously hash-verified actual v5 predictions were reused where the fold assignments matched; stratified and canonical-city references were freshly trained.

A local champion requires positive gain on both random confirmation assignments and city deterioration no greater than 1%. An exploratory portal trial may have mixed results: random mean deterioration must be at most 1%, city at most 3%, and stratified at most 2%. These limits were set before confirmation. Trial order prefers local champions, then the best random confirmation mean, with different families.

Turnover-bin errors and squared-error shares are in `results/segments.csv`; fold/tail counts are in `results/folds.csv`.

**Material tradeoff:** The residual finalist improves the full metric by reducing higher-turnover errors, while worsening the <=40M segment on both random confirmation assignments (about INR 214k–238k RMSE). The tier alternative improves that lower-turnover segment but worsens the full metric. This is why neither the overall local win nor the segment win guarantees a portal gain on different restaurants.

Conditional paired-row bootstrap intervals for the two random confirmation assignments are in `results/conditional_uncertainty.csv`. They reuse fitted predictions and do not account for historical model selection, retraining or test-distribution uncertainty; they were not used to retune the shortlist.

For the residual finalist, the conditional 95% interval is approximately **INR -30,844 to +84,926** and includes zero. The repeated positive comparisons are encouraging, but a generalization improvement is not established conclusively. Treat the export as a portal trial, not a promised better score.

## Experimental files and rank-1 feasibility

### Trial 1: residual_tree_l2, weight 75%

- Status: **local champion**.
- Mean random-confirmation RMSE gain: **INR 24,971.50**.
- Prediction RMS difference from scored v5: **INR 460,050.73**.
- Conditional portal RMSE lower bound: **INR 11,586,671.64**.
- Can mathematically beat the shown 4,796,740.56 leader under the stated assumptions: **False**.
- CSV: `candidates/trial_1/submission.csv`; matching executed notebook in the same folder.
- Fresh notebook: 10 code cells, a complete five-fold comparison and full fitting; maximum prediction difference **INR 0.00000000**.
- Passed permutation, missing/unseen values, target/ID invariance, alignment and 12 malformed-file rejections.

### Trial 2: tiers_d4_shrink300, weight 25%

- Status: **exploratory; mixed local validation**.
- Mean random-confirmation RMSE gain: **INR -3,644.17**.
- Prediction RMS difference from scored v5: **INR 503,434.52**.
- Conditional portal RMSE lower bound: **INR 11,543,287.85**.
- Can mathematically beat the shown 4,796,740.56 leader under the stated assumptions: **False**.
- CSV: `candidates/trial_2/submission.csv`; matching executed notebook in the same folder.
- Fresh notebook: 10 code cells, a complete five-fold comparison and full fitting; maximum prediction difference **INR 0.00000000**.
- Passed permutation, missing/unseen values, target/ID invariance, alignment and 12 malformed-file rejections.

The bound uses `RMSE(new) >= RMSE(v5) - RMS(new_prediction - v5_prediction)`. It assumes the same 500 restaurants, the supplied v5 score and ordinary unweighted RMSE as stated in the rules. A candidate must differ by more than INR 7,249,981.81 in RMS terms to be capable of beating the shown leader. Meeting that condition would not prove accuracy. Small changes may still improve the current score.

## Colab, reproducibility and submission

- These experiments and candidate-notebook checks ran locally on CPU; no Colab account was accessed.
- `V7_Research_Colab.ipynb` is a portable driver for the same bounded research and is not represented as cloud-executed.
- Candidate notebooks embed all predictor code, require only the four official CSVs, and were executed in fresh local processes.
- Fold checkpoints have input/code/configuration/fold/file hashes and can resume. Use optional Google Drive storage in Colab to survive runtime deletion; local runtime storage alone is temporary.
- Verified the embedded research source in an isolated folder without earlier projects, checked valid checkpoint reuse, rejected changed configurations and corrupted contents, and recovered a write missing its completion metadata.
- No paid resources are needed for this implementation. Free Colab resources and lifetime vary; the screenshot estimate is not a guarantee.
- Upload trial CSVs separately and record returned scores in `portal_trials.csv`. Keep the lowest observed score, including v5.
- Submit the one-time coding assignment only after choosing the best scored CSV, using its matching executed notebook.
- No portal uploads were performed by these scripts. The user subsequently reported the Trial 1 result above. First place is not achieved by that result. All validation tables and candidate artifacts below remain the original pre-upload measurements and files.

Every validation assignment reuses labels examined in earlier rounds. The results are robustness evidence, not an untouched external test or an unbiased evaluation of the complete historical search.

Primary references: [CatBoost probability prediction](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict_proba), [Google Colab runtime/storage limits](https://research.google.com/colaboratory/faq.html).
