# V13 measured report

**Retain V9. No new V13 submission is recommended.**

Best observed portal RMSE: V9, 11,904,515.9864996. Validation results are not portal scores.
Three fixed treatments: Q99 and Q995 cap training labels at fold-local percentiles; SQ2 uses the square-root predictive second moment. Every training row is retained. Every validation metric uses the original uncapped labels.

## Feasibility

```json
{
  "Q99": {
    "status": "PASS",
    "fold_seconds": 121.36219469599996,
    "behavior": {
      "permutation_difference_INR": 0.0,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 0.0,
      "individual_difference_INR": 0.0,
      "missing_unseen_inputs": true
    },
    "fresh_repeat_difference_INR": 0.0,
    "target_metadata": {
      "name": "Q99",
      "training_rows": 2794,
      "units": "INR millions",
      "quantile": 0.99,
      "quantile_method": "linear",
      "threshold_INR": 102535000.00000007,
      "capped_rows": 28
    },
    "training_rows": 2794,
    "validation_rows": 699,
    "engineering_seconds": 681.766265037
  },
  "Q995": {
    "status": "PASS",
    "fold_seconds": 125.67824645899987,
    "behavior": {
      "permutation_difference_INR": 0.0,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 0.0,
      "individual_difference_INR": 0.0,
      "missing_unseen_inputs": true
    },
    "fresh_repeat_difference_INR": 0.0,
    "target_metadata": {
      "name": "Q995",
      "training_rows": 2794,
      "units": "INR millions",
      "quantile": 0.995,
      "quantile_method": "linear",
      "threshold_INR": 150000000.0,
      "capped_rows": 13
    },
    "training_rows": 2794,
    "validation_rows": 699,
    "engineering_seconds": 686.2797140139999
  },
  "SQ2": {
    "status": "PASS",
    "fold_seconds": 125.15698242200006,
    "behavior": {
      "permutation_difference_INR": 0.0,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 0.0,
      "individual_difference_INR": 0.0,
      "missing_unseen_inputs": true
    },
    "fresh_repeat_difference_INR": 0.0,
    "target_metadata": {
      "name": "SQ2",
      "training_rows": 2794,
      "units": "sqrt(INR millions)"
    },
    "training_rows": 2794,
    "validation_rows": 699,
    "engineering_seconds": 684.9002949790001
  }
}
```

Screened settings: Q99, Q995, SQ2.

## Development

Each new model was blended with V9 at 25%, 50%, 75%, 100%.

| Setting | Weight | Gain 42, INR | Gain 71, INR | Mean relative gain |
|---|---:|---:|---:|---:|
| Q99 | 25% | 1,354.86 | -7,161.17 | -0.01471% |
| Q99 | 50% | -4,047.62 | -21,247.41 | -0.06409% |
| Q99 | 75% | -16,201.89 | -42,243.92 | -0.14810% |
| Q99 | 100% | -35,095.50 | -70,128.71 | -0.26663% |
| Q995 | 25% | 1,219.39 | -5,585.23 | -0.01106% |
| Q995 | 50% | -3,873.43 | -17,547.26 | -0.05428% |
| Q995 | 75% | -15,273.59 | -35,874.50 | -0.12960% |
| Q995 | 100% | -32,970.18 | -60,549.27 | -0.23697% |
| SQ2 | 25% | 5,193.64 | -3,464.43 | 0.00438% |
| SQ2 | 50% | 3,590.06 | -14,037.57 | -0.02647% |
| SQ2 | 75% | -4,809.08 | -31,708.00 | -0.09253% |
| SQ2 | 100% | -19,995.11 | -56,456.72 | -0.19372% |

## Confirmation and decision

Retain V9: no development candidate qualified.
Confirmation and bootstrap qualification were not performed because no development candidate qualified. No new V13 CSV or executed V13 submission notebook is claimed. Original scored V9 files are preserved byte-for-byte.

Development requires improvement on both assignments and mean relative gain >=0.20%. One finalist is frozen before confirmation. Confirmation requires positive gains on all three random assignments, mean relative gain >=0.20%, a positive lower 95% paired-row gain bound from 5,000 resamples, and at most 1% deterioration on either city or stratified checks. Previously examined labels provide conditional robustness evidence, not an untouched test.

## Compute and reproduction

Conservative task elapsed charge: 1.8737 hours; approved maximum eight hours. Free T4 only. No portal upload was performed.
V7 uses the user-owned original-Mac fitted model independently verified in V9. Fresh Linux V7 retraining is not claimed. Both required TabPFN components train afresh in a qualified standalone notebook; expected CSVs are numerical verification witnesses only.
TabPFN v2 attribution: Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. Checkpoint revision, SHA256, packages, target thresholds, sources and fold identities accompany this report.

Reconfirm the portal deadline and timezone before the single manual trial. Record its returned RMSE against the exact CSV hash, retain the lower observed score, and keep the one-time coding upload pending the final choice. No rank is promised.
