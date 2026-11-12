# Restaurant turnover: fourth improvement round

**Student register number:** submission  
**Student name:** submission  
**Team name:** submission

## Actual hackathon results

These values come from the supplied screenshots, not from local validation.

| Submission | Portal RMSE | Rank shown |
|---|---:|---:|
| Original v1 | 12,204,615.6438253 | 7 before the latest upload |
| Successful v3, Oct 12 at 10:22 PM | 12,094,600.3216516 | 3 |
| Leader shown, Aathan | 11,946,659.0702364 | 1 |
| New v4 candidate | **Unknown until uploaded** | Unknown |

The successful upload reduced RMSE by **110,015.32**. It still needs a reduction greater than **147,941.25**, or about **1.223%**, to beat the leader shown if that leader score is unchanged. No rank guarantee is possible from local validation.

## What was changed

The round screened **24 settings**: 12 numeric-model parameter variants, 10 feature representations including the base comparison, and two feature-based training-weight variants. It compared **78 coarse blends**, including two equal-weight family combinations, using five-fold seed 42 only. The selected formula was frozen before the checks below:

- 25% of the exact successful v3 ensemble.
- 37.5% parameter_borders32, averaging seeds [1042, 2042].
- 37.5% feature_categorical_ratings, averaging seeds [1042, 2042].

The numeric variant uses 32 candidate split borders. The categorical variant treats 18 integer rating fields as categories. Both use depth 5, 850 trees, learning rate 0.04, Huber delta INR 40 million, L2 10 and Bernoulli subsampling 0.7. The 18 rating fields had no unseen test categories or string-format collisions. New components use model seeds 1042/2042, plus the fold index during validation.

Development RMSE changed from **19,839,027.54** to **19,823,848.42**. This score was used for selection and is optimistic evidence. We did not keep searching or retune the weights after seeing confirmation scores.

## Same-fold local RMSE comparison

All 3,493 original training targets are scored in original INR units. Lower is better. A positive reduction means improvement.

| Validation assignment | Successful v3 model | New v4 candidate | RMSE reduction |
|---|---:|---:|---:|
| 1701 — new split | 19,867,153.71 | 19,874,296.41 | -7,142.69 |
| 1711 — new split | 19,811,265.64 | 19,812,900.49 | -1,634.86 |
| 71 | 19,870,083.61 | 19,857,784.25 | +12,299.36 |
| 2026 | 19,823,207.78 | 19,818,699.61 | +4,508.17 |
| 303 | 19,887,896.10 | 19,878,783.03 | +9,113.07 |
| 909 | 19,867,479.32 | 19,847,565.72 | +19,913.60 |
| City groups — stress | 20,129,673.34 | 20,149,537.41 | -19,864.08 |
| Mean of six shuffled assignments | 19,854,514.36 | 19,848,338.25 | +6,176.11 |
| Mean of the two new assignments | 19,839,209.68 | 19,843,598.45 | -4,388.77 |

The candidate improves **4 of six** shuffled assignments. City groups are a separate stress test and are excluded from their average. The city comparison uses the existing canonical city aliases; other spelling variants may still represent the same city, so it is not a perfect geographic holdout.

## Interpretation and limitations

The candidate does not improve both the overall and newly reserved split averages. Keep the portal-confirmed v3 file as the preferred model. The v4 file is an experimental alternative, not a verified improvement.

The conditional row-bootstrap 95% interval for the RMSE reduction is **[-13,527.06, 25,867.89]**. It includes zero, so an incremental improvement is not established by this check. The interval resamples existing prediction errors; it does not refit or repeat the whole search.

All split assignments reuse the same labeled restaurants, including labels used in earlier development. Seeds 1701/1711 were reserved during this round, but this is not an untouched external test or nested evaluation of all earlier searches. The portal evaluates different restaurants. Local and portal RMSE therefore need not match, and their difference cannot be subtracted from a new local score to predict rank.

Validation rows and targets were never deleted, clipped or log-transformed. Huber loss and the inherited training-only cap affect model training only. No hidden targets or reconstructed labels were used. Test feature missingness informed two explicitly evaluated weighting hypotheses, which were not selected.

## Files, execution and checks

- `submission.csv`: exactly 500 records, two exact headers, original test restaurant IDs and row order, finite nonnegative predictions, no index column.
- `submission.ipynb`: self-contained Colab notebook with the same frozen model and checks. Student identity appears on its cover and in filenames; row IDs remain restaurant IDs.
- The original successful v3 files are preserved in `rank3_confirmed_backup/` with their hashes and screenshot score in `confirmed_portal_result.json`.
- The independent full fit reproduces v3's submitted predictions to numerical precision before constructing v4.
- The executed notebook transparently reuses hash-verified fold predictions, recomputes RMSE/figures, and reruns all full-data fits and behavior/export checks. Colab defaults to retraining validation folds. A separate uncached five-fold notebook run checks that default path.
- Invalid headers, row count/order, extra index, duplicates, student IDs, NaN, infinity and negative predictions are tested for rejection. Predictions are also checked under row permutation and missing/unseen feature values.

**Completed execution checks:** all 10 notebook code cells ran with no saved errors and produced two figures. The independent file checker rejected 12 deliberately invalid submissions. The separate uncached five-fold run on split 1701 passed with **0.0 INR maximum prediction difference** for both the confirmed model and the new candidate. Execution logs and machine-readable results are in `results/`. Candidate portal accuracy is unmeasured.

### Download the intended version

- [Confirmed best CSV — v3](../submission.csv) and [matching executed notebook](../submission.ipynb): these are the preserved model files associated with the observed rank-3 score. The notebook was executed before that portal result; its original pending-result statements are historical.
- [Experimental v4 CSV](../improvement_v4/submission.csv) and [matching executed Colab notebook](../improvement_v4/submission.ipynb): these contain this round's unscored candidate and its complete comparison results. Both versions use the required filenames in separate folders.

## Submission

If you choose to test this experimental v4 CSV, upload it to **assignment M.Tech. MLA Hackathon -Restaurant Turnover Prediction** and keep its returned score with this exact version. Until a better portal result is observed, retain the confirmed v3 files. Submit the executed notebook matching your final chosen model to **LLJ_Oct_2026**; the supplied instructions say this assignment allows one submission. A filename change does not alter the portal registration name.
