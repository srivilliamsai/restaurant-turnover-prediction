# V10 measured report

**V10 passed the local qualification gates but did not improve the portal score. Retain scored V9.**

Current observed portal best: V9, RMSE 11,904,515.9864996. The latest supplied screenshot shows rank 11 and confirms the previous best was retained. V10 trial 1 scored 11,925,736.5011738: worse by 21,220.5146742 (0.1783%). The declared one-trial allowance is used. See portal_result.json and portal_result.png; no additional V10 upload is recommended.

All 3,493 original targets were retained. Six fixed TabPFN feature settings and 24 fixed replacement blends were evaluated. V7 retains a 50% share; replacement weights alter only the TabPFN half. No target filters, external labels, fitted calibrator or prediction rounding were used.

For P settings, input-removal invariance is verified for the new TabPFN component. The frozen V7 and original T9 shares retain their original features; the full blend is not claimed to ignore Food Rating/Order Wait Time globally.

A numerical preflight found INR 1.295133 batch-size dependence in native float32 inference. The required batch-invariance check therefore uses fixed 500-row internal batches with a frozen training-feature padding row; padded outputs are discarded. All V9 TabPFN validation controls are refitted on the same folds with that adapter, and original OOF witnesses remain preserved. The full 500-row submission uses its original shape and must still reproduce the scored V9 within INR 0.01.

## Development: best blend for each setting

| Setting | Replacement weight | Gain seed 42 | Gain seed 71 |
|---|---:|---:|---:|
| C | 100% | 4,925.06 | 3,600.48 |
| S | 25% | -3,872.89 | -321.13 |
| C+S | 25% | -1,640.49 | 2,013.65 |
| C+S+G | 50% | -784.04 | 7,124.06 |
| P | 75% | -1,277.07 | 2,827.31 |
| C+S+P | 25% | -1,960.49 | 1,372.81 |

Every blend, including failures, is retained in screening.csv. Component-alone, fold, segment and row-concentration diagnostics accompany the report.

## Frozen finalist

C; replacement weight 100%.

| Check | V9 RMSE | V10 RMSE | Gain |
|---|---:|---:|---:|
| 48211 | 19,743,626.59 | 19,736,521.92 | 7,104.66 |
| 48212 | 19,816,597.34 | 19,812,530.04 | 4,067.30 |
| 48213 | 19,711,188.54 | 19,710,108.76 | 1,079.78 |
| strat:27421 | 19,726,481.46 | 19,724,283.49 | 2,197.96 |
| city | 20,061,150.77 | 20,042,973.91 | 18,176.86 |

Conditional 5,000-resample paired-row mean-gain interval: [-6,546.86, 15,187.14] INR. It excludes selection, retraining and hidden-test distribution uncertainty. Confirmation labels have been examined historically.

## Reproduction and provenance

The frozen V7 state was independently freshly fitted and verified on its original Mac platform. Fresh Linux V7 fitting is not claimed to reproduce it. The delivery freshly fits all required TabPFN components in a separate standalone notebook execution. Expected predictions are verification witnesses only; they are not features or inference lookups.

Fresh standalone execution passed: maximum exported-prediction difference was INR 0.000000007451, below INR 0.01. V9 was independently reproduced within INR 0.000000007451. Exact dependency, source, input, checkpoint and output hashes are recorded. Only free T4 compute was requested. Preserve the original V9 files and keep the one-time coding upload pending the final choice.

The Gemini source inventory records every inspected export without executing those scripts. Their large ensemble was not adopted: its globally precomputed target encodings compromise validation, its weights optimize a restricted target range, and some drafts are incomplete.

TabPFN attribution: Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. No first-place score is promised. Deadline timezone remains to be verified before a portal upload.

## Compute accounting

Final recorded global charge: 16.925 hours under the authorized 24-hour total cap. This includes 10.686 hours charged conservatively for an unobserved disconnected interval; actual computation during that interval is unknown. Two detached free-runtime attempts ended without a completed screen fold. A later interrupted foreground run preserved 26 completed folds. Allocation retries and the package-install recovery were charged; interrupted pilot time is retained. Subsequent actual training uses foreground notebook execution with supervised limits. The final compute_budget.json is authoritative.
