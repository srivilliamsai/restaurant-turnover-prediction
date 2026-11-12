# V9 recovered delivery

**V9 is the current observed portal best: RMSE 11,904,515.9864996; rank 9 in the user-provided October 27 screenshot.**

The frozen ensemble is 50% your own scored V7 model and 50% raw-target TabPFN v2, V7 features, 16 estimators. All 3,493 original training targets were retained. No paid compute was used. The user uploaded V9; the agent performed no portal upload.

## Reproduction resolution

Fresh V7 fitting on the original Mac reproduced the scored file within INR 0.000000007451. Fresh fitting in Colab had differed by up to INR 427,996.18; matching input and target hashes ruled out changed data. Most component predictions matched exactly, while one V5 component and the nested residual fits differed. The precise low-level numerical cause was not established.

The delivered notebook embeds the user-owned model from that separately verified fresh Mac fit. Loading its fitted state in Colab preserves V7 predictions within INR 0.000000022352. It does not claim that V7 is retrained identically on Linux. V7 training source and provenance are included; TabPFN trains afresh on every notebook execution. No expected prediction vector is used as a predictor.

The three random confirmation assignments were recomputed using fresh Mac V7 fits and the unchanged, hash-verified TabPFN predictions. Stratified/city checks use their verified original-platform V7 snapshots. The frozen choice and gates were unchanged.

## Development

Nine settings and 36 blend configurations were evaluated. Each row below is the best development blend for one setting; every configuration, including failures, is retained in development_all_36.csv.

| Setting | Mode | Candidate weight | Gain 42 | Gain 71 |
|---|---|---:|---:|---:|
| tabpfn_raw_v7 | ensemble16 | 50% | 84,572.12 | 57,759.32 |
| tabpfn_raw_v7 | plain | 50% | 83,488.59 | 54,611.07 |
| tabpfn_raw_compact | plain | 50% | 70,435.76 | 40,583.81 |
| tabpfn_sqrt_v7 | plain | 25% | 41,896.38 | 27,800.46 |
| tabpfn_sqrt_compact | plain | 25% | 35,318.95 | 22,784.40 |
| ebm_sqrt_i10 | plain | 25% | 1,580.09 | -9,985.79 |
| ebm_sqrt_i0 | plain | 25% | -4,329.31 | -17,678.07 |
| ebm_raw_i0 | plain | 25% | -13,171.74 | -30,574.60 |
| ebm_raw_i10 | plain | 25% | -21,279.89 | -31,940.86 |

## Measured confirmation comparisons

V7 was fitted on its original platform for these comparisons. No candidate setting or blend weight changed. These reused labels provide conditional robustness evidence, not an untouched external test.

| Assignment | V7 RMSE | V9 RMSE | Gain |
|---|---:|---:|---:|
| 48211 | 19,835,389.18 | 19,743,626.70 | 91,762.48 |
| 48212 | 19,930,476.67 | 19,816,597.32 | 113,879.35 |
| 48213 | 19,790,513.92 | 19,711,188.53 | 79,325.39 |
| strat:27421 | 19,788,419.32 | 19,726,481.41 | 61,937.91 |
| city | 20,113,476.99 | 19,964,901.25 | 148,575.74 |

Conditional 5,000-resample paired-row 95% interval for mean random-assignment RMSE gain: [35,752.50, 159,172.10] INR. This excludes search, retraining and distribution uncertainty.

All three random gains are positive. Stratified and city results satisfy the maximum 1% deterioration gate. No alternative EBM candidate qualified.

## Error concentration

- 48211: the ten largest positive row contributions represent 50.1% of net squared-error reduction; the >80M group contributes -2.3%. Negative values mean deterioration.
- 48212: the ten largest positive row contributions represent 56.4% of net squared-error reduction; the >80M group contributes -4.2%. Negative values mean deterioration.
- 48213: the ten largest positive row contributions represent 73.5% of net squared-error reduction; the >80M group contributes -48.5%. Negative values mean deterioration.
- strat:27421: the ten largest positive row contributions represent 57.8% of net squared-error reduction; the >80M group contributes -50.5%. Negative values mean deterioration.
- city: the ten largest positive row contributions represent 45.2% of net squared-error reduction; the >80M group contributes 49.7%. Negative values mean deterioration.

All segment results (≤40M, 40–80M, >80M) and fold results accompany this report. Segment diagnostics never determine inference-time rules.

## Submission and limits

CSV: submission/submission.csv

Matching executed notebook: submission/submission.ipynb

Fresh standalone notebook maximum prediction difference: INR 0.000000014901. Both components passed row-permutation, ID/target invariance, missing/unseen-input and malformed-CSV checks. The final CSV has 500 unique IDs in original order, exact headers, and finite nonnegative values.

RMS prediction distance from scored V7 is INR 1,362,609.87. Assuming the stated ordinary RMSE on all the same 500 rows, the reverse triangle inequality bounds this candidate’s portal RMSE below by INR 10,661,706.67. Consequently this particular candidate cannot beat the supplied 4,796,740.56 leader score under those assumptions. The bound is not a forecast of its actual score.

Upload only the CSV for a portal trial, record the returned RMSE against its exact hash, and keep whichever observed file scores lower. Preserve the one-time coding upload until the final choice. Deadline displayed in the supplied portal: October 29, 11:59PM; timezone remains unconfirmed.

TabPFN uses the official Prior Labs general-purpose v2 regression checkpoint; attribution: Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. No competitor predictions or competition-trained models are used.


## Observed portal result — October 27

The user supplied a result screenshot showing RMSE **11,904,515.9864996**, previous best **12,024,316.5355221**, and rank **9**. This improves RMSE by **119,800.55 (0.9963%)**. V9 is now the recommended current submission.

CSV SHA256: `5a384dc3d8bf58c4e29f0d6cac646aa33afe22bdcdd186cccd9758342769bbd4`.

The score-to-file association follows the preceding V9 upload instructions and user response; the screenshot itself does not show the filename or hash. The CSV and executed notebook remain byte-identical to the verified delivery and are preserved in `scored_v9_backup`. No new model fitting or parameter selection was performed after observing this score. The executed notebook retains its pre-submission narrative as a dated execution snapshot.
