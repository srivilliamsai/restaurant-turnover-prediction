# V12 measured report

**Retain V9. No new V12 submission is recommended.**

Best observed portal RMSE remains V9: 11,904,515.9864996. No V12 portal score is inferred from validation.
All 3,493 targets and the original 500 test IDs/order are retained. The forty source predictors are unchanged from V9. IDs are never predictors.

## Feasibility

{
  "R1": {
    "status": "PASS",
    "fold_seconds": 54.86512459000005,
    "repeat_difference_INR": 0.0,
    "behavior": {
      "permutation_difference_INR": 0.0,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 0.0,
      "individual_difference_INR": 0.0,
      "missing_unseen_inputs": true
    },
    "peak_gpu_allocated_bytes": 81164288,
    "training_rows": 2794,
    "validation_rows": 699,
    "engineering_seconds": 96.76553116900004
  },
  "R4": {
    "status": "PASS",
    "fold_seconds": 160.21270178299994,
    "repeat_difference_INR": 0.0,
    "behavior": {
      "permutation_difference_INR": 0.0,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 0.0,
      "individual_difference_INR": 0.0,
      "missing_unseen_inputs": true
    },
    "peak_gpu_allocated_bytes": 81164288,
    "training_rows": 2794,
    "validation_rows": 699,
    "engineering_seconds": 320.85325269199996
  },
  "T32": {
    "status": "PASS",
    "fold_seconds": 245.82442483900002,
    "repeat_difference_INR": 0.0,
    "behavior": {
      "permutation_difference_INR": 0.0,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 0.0,
      "individual_difference_INR": 0.0,
      "missing_unseen_inputs": true
    },
    "peak_gpu_allocated_bytes": 373881856,
    "training_rows": 2794,
    "validation_rows": 699,
    "engineering_seconds": 1505.69808385
  },
  "T4x16": {
    "status": "PASS",
    "fold_seconds": 460.5184873770004,
    "repeat_difference_INR": 0.0,
    "behavior": {
      "permutation_difference_INR": 0.0,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 0.0,
      "individual_difference_INR": 0.0,
      "missing_unseen_inputs": true
    },
    "peak_gpu_allocated_bytes": 507655168,
    "training_rows": 2794,
    "validation_rows": 699,
    "engineering_seconds": 2733.8903738259996
  }
}

Feasible settings: T32, T4x16, R1, R4

## Development

T settings preserve the V7 contribution: .5*V7+.5*((1-w)*T9+w*Tnew). R settings use (1-w)*V9+w*RealMLP. Only w=25/50/75/100% was tested.

| Setting | Weight | Gain 42, INR | Gain 71, INR | Mean relative gain |
|---|---:|---:|---:|---:|
| T32 | 25% | 1,124.42 | 329.40 | 0.00368% |
| T32 | 50% | 2,148.33 | 565.29 | 0.00688% |
| T32 | 75% | 3,071.71 | 707.65 | 0.00958% |
| T32 | 100% | 3,894.56 | 756.49 | 0.01179% |
| T4x16 | 25% | 472.50 | 494.25 | 0.00245% |
| T4x16 | 50% | 865.87 | 916.80 | 0.00452% |
| T4x16 | 75% | 1,180.11 | 1,267.63 | 0.00620% |
| T4x16 | 100% | 1,415.21 | 1,546.74 | 0.00751% |
| R1 | 25% | -12,877.08 | -16,304.18 | -0.07394% |
| R1 | 50% | -200,469.89 | -190,094.44 | -0.98967% |
| R1 | 75% | -557,932.78 | -517,316.49 | -2.72462% |
| R1 | 100% | -1,076,512.83 | -990,376.72 | -5.23738% |
| R4 | 25% | -34,340.62 | 28,623.68 | -0.01449% |
| R4 | 50% | -245,727.78 | -35,928.43 | -0.71372% |
| R4 | 75% | -628,647.64 | -192,750.78 | -2.08142% |
| R4 | 100% | -1,173,676.34 | -439,691.56 | -4.08826% |

## Frozen confirmation

No development finalist met the fixed gates. Confirmation, bootstrap intervals, new CSV export and fresh V12 numerical reproduction were not performed. Original scored V9 files are preserved byte-for-byte.

Qualification: improve both development assignments with mean relative gain >=0.20%; then improve all three random confirmations with mean gain >=0.20%, positive paired-row 95% lower gain bound (5,000 draws), and city/stratified deterioration <=1%. Every failed comparison is reported. Labels are historically examined; intervals are conditional robustness evidence.

## Reproduction and compute

V7 uses the independently verified original-Mac fitted state. Fresh Linux V7 refitting is not claimed to reproduce scored V7. TabPFN and any new RealMLP components train afresh in each qualified standalone notebook. Expected CSVs are read only after inference for verification.
TabPFN v2 attribution: Prior Labs, Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. RealMLP attribution: Holzmüller, Grinsztajn and Steinwart; official pytabkit 1.7.3. Full parameters, versions and official checkpoint hashes are recorded.
Recorded task charge: 6.5570 hours; user-approved maximum eighteen hours. All earlier V12 charges are retained. Prior round ledgers are unchanged. Free T4 through CLI; original V7 controls on Mac. No paid resources or automatic portal uploads.

Training stops by October 29, 2026 at 18:00 IST; artifact deadline 21:00 IST; trial target 22:00 IST. Working final deadline 23:59 IST, portal timezone still requires verification. Keep one-time coding upload pending final model choice.

At most two independently qualified portal trials, in frozen development order. Record actual returned scores against exact CSV hashes, keep the lowest observed score, and do not tune against portal feedback. No first-place result is promised.
