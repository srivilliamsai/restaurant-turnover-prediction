# V14 measured report

**One candidate qualified for a manual portal trial.**

Best observed portal RMSE: V9, 11,904,515.9864996. Validation results are not portal scores.
Two predeclared backbone tests: TabICLv2 I8 and actual TabPFN-3.5 Base P35_8. Unchanged forty-column V9 features; raw targets in INR millions; every original target retained.

## Engineering and feasibility

```json
{
  "I8": {
    "status": "PASS",
    "fold_seconds": 11.559364498000036,
    "behavior": {
      "permutation_difference_INR": 7.62939453125,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 3.814697265625,
      "individual_difference_INR": 0.0,
      "rmse_difference_INR": {
        "permutation": 0.027425337582826614,
        "batch": 0.030591782182455063,
        "individual": 0.0
      },
      "rows_checked": 699,
      "individual_probes": 12,
      "missing_unseen_inputs": true,
      "behavior_tolerance_INR": 10.0,
      "rmse_tolerance_INR": 1.0
    },
    "fresh_repeat_difference_INR": 0.0,
    "target_metadata": {
      "name": "I8",
      "training_rows": 2794,
      "units": "raw INR millions",
      "checkpoint_sha256": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
      "model_class": "tabicl._sklearn.regressor.TabICLRegressor"
    },
    "peak_gpu_allocated_bytes": 2029224448,
    "training_rows": 2794,
    "validation_rows": 699,
    "engineering_seconds": 30.271495617000028
  },
  "P35_8": {
    "status": "PASS",
    "fold_seconds": 22.19464035200002,
    "behavior": {
      "permutation_difference_INR": 0.0,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 0.0,
      "individual_difference_INR": 0.0,
      "rmse_difference_INR": {
        "permutation": 0.0,
        "batch": 0.0,
        "individual": 0.0
      },
      "rows_checked": 699,
      "individual_probes": 12,
      "missing_unseen_inputs": true,
      "behavior_tolerance_INR": 10.0,
      "rmse_tolerance_INR": 1.0
    },
    "fresh_repeat_difference_INR": 0.0,
    "target_metadata": {
      "name": "P35_8",
      "training_rows": 2794,
      "units": "raw INR millions",
      "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
      "model_class": "tabpfn.regressor.TabPFNRegressor",
      "architecture_classes": [
        "tabpfn.architectures.tabpfn_v3_5.TabPFNV3p5"
      ],
      "configuration_classes": [
        "tabpfn.architectures.tabpfn_v3_5.TabPFNV3p5Config"
      ]
    },
    "peak_gpu_allocated_bytes": 7042150400,
    "training_rows": 2794,
    "validation_rows": 699,
    "engineering_seconds": 247.54860044100008
  }
}
```

Screened settings: I8, P35_8.
Budget exclusions: {}

Same-layout fresh fits and CSV reproduction require INR 0.01. Permutation, split-batch and individual-row behavior checks permit at most INR 10 per prediction and INR 1 RMSE change on the labeled pilot. Feature target/ID invariance remains exact. No output rounding is used.

## Development

Route A: (1-w)*V9+w*M. Route B: .5*V7+.5*((1-w)*T9+w*M). Weights: 25%, 50%, 75%, 100%. Both routes share the same complete component folds.

| Setting | Route | Weight | Gain 42, INR | Gain 71, INR | Mean relative gain |
|---|---|---:|---:|---:|---:|
| I8 | A | 25% | 19,976.62 | 10,093.75 | 0.07620% |
| I8 | A | 50% | 11,826.19 | -12,927.00 | -0.00279% |
| I8 | A | 75% | -24,416.49 | -68,946.75 | -0.23657% |
| I8 | A | 100% | -88,597.29 | -157,686.71 | -0.62406% |
| I8 | B | 25% | 9,038.50 | 8,240.01 | 0.04378% |
| I8 | B | 50% | 10,690.94 | 7,816.79 | 0.04690% |
| I8 | B | 75% | 4,955.46 | -1,269.10 | 0.00934% |
| I8 | B | 100% | -8,161.50 | -19,005.71 | -0.06884% |
| P35_8 | A | 25% | 41,921.73 | 41,984.54 | 0.21261% |
| P35_8 | A | 50% | 69,344.69 | 69,709.54 | 0.35235% |
| P35_8 | A | 75% | 82,208.17 | 83,114.63 | 0.41892% |
| P35_8 | A | 100% | 80,483.59 | 82,170.50 | 0.41216% |
| P35_8 | B | 25% | 18,323.48 | 21,824.63 | 0.10173% |
| P35_8 | B | 50% | 32,708.06 | 39,713.70 | 0.18351% |
| P35_8 | B | 75% | 43,145.09 | 53,656.47 | 0.24529% |
| P35_8 | B | 100% | 49,628.30 | 63,644.54 | 0.28702% |

## Confirmation and decision

One qualified candidate for manual portal trial.

```json
{
  "choice": {
    "name": "P35_8",
    "route": "A",
    "weight": 0.75,
    "effective_new_share": 0.75,
    "rmse_42": 19649033.55616177,
    "gain_42": 82208.17207151651,
    "relative_gain_42": 0.004166396276717114,
    "rmse_71": 19649905.454410896,
    "gain_71": 83114.62613611668,
    "relative_gain_71": 0.004211956699828823,
    "mean_relative_gain": 0.004189176488272968
  },
  "qualified": true,
  "metrics": {
    "48211": {
      "v9_rmse": 19743626.58666369,
      "candidate_rmse": 19661515.847807977,
      "gain": 82110.73885571212,
      "relative_gain": 0.004158847843646709
    },
    "48212": {
      "v9_rmse": 19816597.33709491,
      "candidate_rmse": 19672213.5596034,
      "gain": 144383.7774915099,
      "relative_gain": 0.007286002487482364
    },
    "48213": {
      "v9_rmse": 19711188.541631036,
      "candidate_rmse": 19619589.028592434,
      "gain": 91599.51303860173,
      "relative_gain": 0.0046470821810231724
    },
    "strat:27421": {
      "v9_rmse": 19726481.457054287,
      "candidate_rmse": 19663361.754943263,
      "gain": 63119.70211102441,
      "relative_gain": 0.0031997445793077554
    },
    "city": {
      "v9_rmse": 20061150.77388296,
      "candidate_rmse": 19924168.53482659,
      "gain": 136982.23905636743,
      "relative_gain": 0.006828234362043712
    }
  },
  "uncertainty": {
    "resamples": 5000,
    "seed": 8319,
    "ci95_INR": [
      35922.47204564797,
      181795.21567397917
    ],
    "interpretation": "Conditional paired-row robustness interval; excludes selection, retraining and hidden-test shift."
  }
}
```

The development minimum is a positive gain on both assignments and mean relative gain >=0.20%. Confirmation requires positive gains on all three random assignments, mean relative gain >=0.20%, a positive lower 95% paired-row gain bound from 5,000 resamples, and city/stratified deterioration <=1%. All labels have been examined historically; this is conditional robustness evidence, not an untouched test.

## Reproduction, compute and submission

Conservative elapsed charge: 1.0162 hours under the separate eight-hour cap. Free T4 only; no portal upload.
The V7 share uses the independently verified original-Mac fitted state. Identical Linux V7 retraining is not claimed. A qualified notebook freshly fits the V9 TabPFN and selected new backbone; expected predictions are verification witnesses only.
Original inputs, baseline provenance, packages, sources, official checkpoint hashes and all failures accompany this report. TabPFN: Prior Labs, Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. TabICLv2: Qu, Holzmüller, Varoquaux and Le Morvan; official soda-inria/tabicl implementation.

After every gate passes, at most one user-uploaded portal trial is recommended. Reconfirm the deadline timezone, record the exact CSV hash and actual returned RMSE, retain the lowest observed score, and keep the one-time coding upload pending. No rank is promised.
