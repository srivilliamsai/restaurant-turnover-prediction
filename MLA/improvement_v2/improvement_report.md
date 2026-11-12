# Restaurant turnover — model improvement report

Student: submission | Register number: submission | Team: submission

## Result

The new candidate reduced mean RMSE across three fixed shuffled five-fold checks from 19,937,841.85 to 19,877,899.73, a reduction of 59,942.12 (0.301%).

The old CSV has a portal-reported RMSE of 12,204,615.6438253 and rank 3. The new CSV has not yet been scored on the portal. The reference leader score was 11,946,659.0702364. First place cannot be confirmed or promised from local validation.

## Equal-condition validation comparison

| Validation | Rank-3 model | New candidate | RMSE reduction |
|---|---:|---:|---:|
| Shuffled 5-fold seed 71 | 19,956,845.01 | 19,891,238.91 | 65,606.11 |
| Shuffled 5-fold seed 2026 | 19,886,765.22 | 19,836,015.51 | 50,749.71 |
| Shuffled 5-fold seed 303 | 19,969,915.32 | 19,906,444.77 | 63,470.55 |
| City GroupKFold | 20,227,279.16 | 20,092,949.44 | 134,329.71 |

The conditional 95% case-bootstrap interval for the RMSE reduction is [13,058, 107,632]. This resamples validation errors; it does not include every source of model or distribution uncertainty.

## What changed

- Compared 38 settings across CatBoost, LightGBM, XGBoost, Extra Trees, histogram boosting, Ridge, spline Ridge and SVR.
- Combined ten retained predictors with nonnegative weights summing to one; weights were fitted using development out-of-fold predictions with regularization.
- Added date, social, rating interaction and missingness features; tested dropping shifted service and comfort ratings.
- Tested Huber objectives, monotonic constraints and a cap applied only to training targets. Validation targets were never modified or removed.
- Used fixed tree counts and weights on subsequent validation. Seed 303 was reserved until the final candidate was fixed.

## Limits of the evidence

The original notebook's selected validation score (19,854,618) used development-fold early stopping. The table above compares the old and new fixed final-model configurations on identical further splits; these are the relevant comparisons for this change. All splits reuse the same labeled data, so they are robustness estimates, not an untouched external test.

The 33 training targets above INR 100 million account for about 57.5% of the mean-baseline squared error. The upper tail and the service/comfort distribution differences are possible reasons validation and portal scores differ. The hidden test labels are unavailable, so the cause cannot be confirmed.

## Execution and checks

- Executed all 12 Python cells; zero cell errors. Then reran the city stress test with explicit aliases grouped together and refreshed its diagnostics.
- Reproduced the old rank-3 CSV from its model within INR 0.01 tolerance.
- Checked the RMSE formula, input schema, target/ID exclusion, train/validation disjointness, city-group isolation, and one validation prediction per row.
- Checked shuffled prediction-row alignment, unseen/missing feature values, finite nonnegative predictions, 500 unique restaurant IDs in test order, exact headers, 501 CSV lines and CSV round-trip.

## Submission

Upload `submission.csv` to the hackathon module. The new file is a candidate whose hidden-test performance must be measured by the portal. Keep the best observed submission.
Upload the executed `submission.ipynb` under LLJ_Oct_2026 only when your model choice is final. The assignment is described as allowing one submission.
The rank-3 files are preserved in `improvement_v2/rank3_backup`. The student register number belongs in the filenames; the CSV row IDs remain the restaurant IDs from Test_dataset.csv.

The portal screenshots show team name submission. Faculty instructions require a Registernumber_Name registration format; filenames do not update portal registration. Confirm that requirement with the portal settings or Program Manager before the deadline.
