# V15 measured report

**Retain V14: no new V15 submission is recommended.**

Best observed portal RMSE: V14, 11,736,110.7877709. V15 has not been uploaded. Local validation is not a portal score.

This is the separately user-authorized four-hour retry. The original interrupted eight-hour attempt and charges remain preserved in prior_attempt and retry_history.json; all five pilots were repeated.

## Fixed experiment

Five settings: F40-T080, F40-T120, N32-T080, N32-T100 and N32-T120. Exact V14 TabPFN-3.5 checkpoint, eight estimators, seed 10242 and original training targets in INR millions. N32 uses native parsed dates, fold-local categorical vocabularies and original numeric missingness. Automatic temperature calibration is disabled.

Formula: .25*V9 + .75*((1-w)*original_P35 + w*new_component), w in [.25,.5,.75,1]. Baseline is scored V14, never V9.

## Engineering pilots

```json
{
  "status": "PASS",
  "settings": {
    "F40-T080": {
      "status": "PASS",
      "fold_seconds": 28.245101604999945,
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
        "name": "F40-T080",
        "training_rows": 2794,
        "units": "raw INR millions",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "model_class": "tabpfn.regressor.TabPFNRegressor",
        "view": "F40",
        "input_features": 40,
        "temperature": 0.8,
        "date_expansion": true,
        "datetime_dependencies": {
          "skrub": "0.10.1",
          "pyarrow": "23.0.1"
        },
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
      "engineering_seconds": 253.907839459
    },
    "F40-T120": {
      "status": "PASS",
      "fold_seconds": 23.725044841,
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
        "name": "F40-T120",
        "training_rows": 2794,
        "units": "raw INR millions",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "model_class": "tabpfn.regressor.TabPFNRegressor",
        "view": "F40",
        "input_features": 40,
        "temperature": 1.2,
        "date_expansion": true,
        "datetime_dependencies": {
          "skrub": "0.10.1",
          "pyarrow": "23.0.1"
        },
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
      "engineering_seconds": 253.937747499
    },
    "N32-T080": {
      "status": "PASS",
      "fold_seconds": 23.680411176999996,
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
        "name": "N32-T080",
        "training_rows": 2794,
        "units": "raw INR millions",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "model_class": "tabpfn.regressor.TabPFNRegressor",
        "view": "N32",
        "input_features": 32,
        "temperature": 0.8,
        "date_expansion": true,
        "datetime_dependencies": {
          "skrub": "0.10.1",
          "pyarrow": "23.0.1"
        },
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
      "engineering_seconds": 253.423631354
    },
    "N32-T100": {
      "status": "PASS",
      "fold_seconds": 23.534438343000147,
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
        "name": "N32-T100",
        "training_rows": 2794,
        "units": "raw INR millions",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "model_class": "tabpfn.regressor.TabPFNRegressor",
        "view": "N32",
        "input_features": 32,
        "temperature": 1.0,
        "date_expansion": true,
        "datetime_dependencies": {
          "skrub": "0.10.1",
          "pyarrow": "23.0.1"
        },
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
      "engineering_seconds": 253.41487606500004
    },
    "N32-T120": {
      "status": "PASS",
      "fold_seconds": 23.62170980399992,
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
        "name": "N32-T120",
        "training_rows": 2794,
        "units": "raw INR millions",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "model_class": "tabpfn.regressor.TabPFNRegressor",
        "view": "N32",
        "input_features": 32,
        "temperature": 1.2,
        "date_expansion": true,
        "datetime_dependencies": {
          "skrub": "0.10.1",
          "pyarrow": "23.0.1"
        },
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
      "engineering_seconds": 253.32484929199995
    }
  },
  "screen_settings": [
    "F40-T080",
    "F40-T120",
    "N32-T080",
    "N32-T100",
    "N32-T120"
  ],
  "engineering_feasible_settings": [
    "F40-T080",
    "F40-T120",
    "N32-T080",
    "N32-T100",
    "N32-T120"
  ],
  "budget_exclusions": {},
  "screen_forecast_seconds": 1596.48717501,
  "confirmation_forecast_seconds": 917.9658021624982,
  "forecast_fits_budget": true,
  "selection_uses_validation_rmse": false,
  "sources": {
    "core.py": "9f94246bfb16644502ba7107fdcffaa935f43439664926185313e802d06716bb",
    "features.py": "ccf2a22927d7226814dabd4f85f58ad0f107cc217d414bd17528f5ae48b3c89c",
    "modeling.py": "5e5fead43a93870661e51a16d9c5dd8c6cc3a13ff1a719f4da5ff9b8f4e87d76",
    "workflow.py": "662ff06d0146ea9c6db8f6361bd6772e1618a6513ecea924ab0a94ff10ca0815",
    "verify.py": "69442c0d1ac4353ac7fa735f32921a6f2ea3b9aa8231a040d06adaa0da817c6a",
    "delivery.py": "924b23814c0cad4afd927c860a759b4c467ea0143a3137a141de4c1bc6842ed8",
    "fresh_worker.py": "8befec4e89fc1dc580b0d745f0d9a422ce20dba360aabb8b7586d95c4c653b61",
    "baseline.py": "0513655049b5d7c0e4e57b3339596e718779cd7bcbcc49bafd76e14e5bd9dad1",
    "legacy_models.py": "031c25ba5a60cf3e6e4c8247553ee951ff71742c846eb103246140929f6cdfba",
    "legacy_reference.py": "1a24f9f116170ea54bd3242264d5cd2a150c3429e039a50cb11c3112e52247c1",
    "frozen_v9_modeling.py": "c4d948bcf9e1aab5e667283eb82889af44bc57147a0c2a234af6322044b55487",
    "v5_contract.json": "3446412f01d536a31123142b984392216efd29094965500251386eac1814ab69",
    "city_map.json": "539387a0a9f5e143228714dde7679c2848a019a1cbbaa87129b0d4e4c092f7f9",
    "reference_manifest.json": "82417467e331d6e470fd1058c4ea4edb5a47f98021efc822176bd9a04b99bbd5",
    "baseline_registry.json": "53e7ddb7fbfbe08162aa97fac8d5ad9dea4981aab3aa942b3e98ebdb2a3060b7"
  },
  "inputs": {
    "Train_dataset.csv": "ec40ebda74ba0bfbfe1af4417a4266a70eadf6e595500e99e5435e7bdc7ee51f",
    "Test_dataset.csv": "ee4137566adb1c0ce356a8c370e7b8a22f681bde492068efe11346d8d44b4f4d",
    "Data_Dictionary.csv": "2d86c00e27448593564a161197b6639ee1eb90c01fa523b67e7a09d2ef405c5e",
    "Sample_Solution.csv": "9519ba4d51eb2d59b8a71b054a62e1f4d48b72cf92930c31f5e5393a2cd135fb"
  },
  "model_contract_sha256": "d9ed2fdffaf57b2bb5e0a0ac8693cea6c27e7e2c1113bdb74dc359413427c792"
}
```

## Development

| Setting | Replacement weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|---:|---:|
| F40-T080 | 25% | 19,650,938.59 | 19,652,205.27 | -1,905.04 | -2,299.82 | -0.01070% |
| F40-T080 | 50% | 19,658,448.95 | 19,659,894.54 | -9,415.39 | -9,989.08 | -0.04938% |
| F40-T080 | 75% | 19,671,558.20 | 19,672,966.94 | -22,524.64 | -23,061.48 | -0.11600% |
| F40-T080 | 100% | 19,690,255.16 | 19,691,411.75 | -41,221.60 | -41,506.29 | -0.21051% |
| F40-T120 | 25% | 19,655,596.69 | 19,655,584.94 | -6,563.13 | -5,679.49 | -0.03115% |
| F40-T120 | 50% | 19,671,812.62 | 19,670,524.87 | -22,779.06 | -20,619.42 | -0.11043% |
| F40-T120 | 75% | 19,697,657.51 | 19,694,704.17 | -48,623.95 | -44,798.71 | -0.23772% |
| F40-T120 | 100% | 19,733,093.52 | 19,728,088.86 | -84,059.96 | -78,183.40 | -0.41284% |
| N32-T080 | 25% | 19,649,020.98 | 19,649,420.31 | 12.57 | 485.14 | 0.00127% |
| N32-T080 | 50% | 19,655,733.17 | 19,655,394.49 | -6,699.62 | -5,489.04 | -0.03102% |
| N32-T080 | 75% | 19,669,163.24 | 19,667,822.12 | -20,129.68 | -17,916.66 | -0.09681% |
| N32-T080 | 100% | 19,689,297.44 | 19,686,690.96 | -40,263.88 | -36,785.50 | -0.19606% |
| N32-T100 | 25% | 19,646,293.28 | 19,646,438.07 | 2,740.28 | 3,467.38 | 0.01580% |
| N32-T100 | 50% | 19,645,329.98 | 19,644,633.84 | 3,703.57 | 5,271.61 | 0.02284% |
| N32-T100 | 75% | 19,646,143.93 | 19,644,493.22 | 2,889.62 | 5,412.23 | 0.02112% |
| N32-T100 | 100% | 19,648,734.91 | 19,646,016.25 | 298.65 | 3,889.20 | 0.01066% |
| N32-T120 | 25% | 19,652,624.97 | 19,651,999.73 | -3,591.42 | -2,094.27 | -0.01447% |
| N32-T120 | 50% | 19,669,359.50 | 19,666,802.79 | -20,325.94 | -16,897.33 | -0.09472% |
| N32-T120 | 75% | 19,699,203.63 | 19,694,285.98 | -50,170.08 | -44,380.52 | -0.24059% |
| N32-T120 | 100% | 19,742,097.92 | 19,734,396.32 | -93,064.37 | -84,490.87 | -0.45181% |

## Confirmation and decision

Retain V14: no development candidate qualified.
No development finalist qualified. Confirmation, bootstrap qualification, a new V15 CSV and a fresh V15 submission notebook were correctly skipped. Byte-identical scored V14 CSV/notebook are preserved as KEEP_V14 files.

Require positive gains on both development assignments and mean relative improvement >=0.20% against V14. One finalist is frozen before confirmation. Confirmation requires all three random gains positive, mean relative gain >=0.20%, a positive lower paired-row 95% gain bound (5,000 draws, seed 8319), and city/stratified deterioration <=1%. Labels have been historically examined; intervals are conditional robustness evidence, not an untouched test.

## Reproduction and compute

Conservative elapsed charge at packaging: 1.9892 hours under the four-hour retry cap. Free T4 only; no portal upload.
V7 uses the independently verified original-Mac fitted state; fresh Linux V7 retraining is not claimed. Qualified standalone notebooks fit every required TabPFN component afresh. Expected predictions are only verification witnesses after inference.
Fresh same-layout and CSV agreement require INR 0.01. Permutation, split-batch and individual probes permit at most INR 10 per prediction and INR 1 pilot RMSE change. No rounding is used to pass checks.

Original inputs, fold witnesses, source/environment/checkpoint hashes and all measured failures accompany this report. At most one manual V15 portal trial is recommended after complete qualification and reproduction. Keep V14 until an actual lower portal RMSE is returned. The one-time coding upload remains unconfirmed.

Sources: Prior Labs TabPFN 9.2.0, official datetime/temperature documentation; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. See RESEARCH_SOURCES.md.
