# V15 measured report

**V15 execution is incomplete. Retain the scored V14 submission.**

Four engineering pilots passed: F40-T080, F40-T120, N32-T080 and N32-T100. The saved record does not establish completion of N32-T120. The runtime heartbeat became stale; the coordinator preserved completed backups and released its owned T4. The original eight-hour task wall-clock allowance has since elapsed. No V15 development comparison, confirmation or portal trial was completed. This is an execution interruption, not a measured accuracy failure.

Best observed portal RMSE: V14, 11,736,110.7877709. V15 has not been uploaded. Local validation is not a portal score.

## Fixed experiment

Five settings: F40-T080, F40-T120, N32-T080, N32-T100 and N32-T120. Exact V14 TabPFN-3.5 checkpoint, eight estimators, seed 10242 and original training targets in INR millions. N32 uses native parsed dates, fold-local categorical vocabularies and original numeric missingness. Automatic temperature calibration is disabled.

Formula: .25*V9 + .75*((1-w)*original_P35 + w*new_component), w in [.25,.5,.75,1]. Baseline is scored V14, never V9.

## Engineering pilots

```json
{
  "status": "INCOMPLETE",
  "settings": {
    "F40-T080": {
      "status": "PASS",
      "fold_seconds": 27.985125729000174,
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
      "engineering_seconds": 253.82619298500003
    },
    "F40-T120": {
      "status": "PASS",
      "fold_seconds": 23.874477172999832,
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
      "engineering_seconds": 257.61358170699987
    },
    "N32-T080": {
      "status": "PASS",
      "fold_seconds": 24.07377872500001,
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
      "engineering_seconds": 257.3095816330001
    },
    "N32-T100": {
      "status": "PASS",
      "fold_seconds": 23.85474992599984,
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
      "engineering_seconds": 257.343075968
    }
  },
  "screen_settings": [],
  "sources": {
    "core.py": "0ce3258b59e30ed122938da8278f2b43c503d4c5eef0d3da453514c039cfccec",
    "features.py": "ccf2a22927d7226814dabd4f85f58ad0f107cc217d414bd17528f5ae48b3c89c",
    "modeling.py": "5e5fead43a93870661e51a16d9c5dd8c6cc3a13ff1a719f4da5ff9b8f4e87d76",
    "workflow.py": "a06f613812ad50dc851dc7f5fc49864ff77f57147edd175066a4ce8de8108d6c",
    "verify.py": "69442c0d1ac4353ac7fa735f32921a6f2ea3b9aa8231a040d06adaa0da817c6a",
    "delivery.py": "b3c186d643c5e6549bc26f4a4353e5667e432a6af22ba2c96cf6698c4414a923",
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
  }
}
```

## Development

| Setting | Replacement weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|---:|---:|

## Confirmation and decision

Retain V14: V15 stopped before a complete development/confirmation decision.
Development screening never started, so no V15 accuracy result or qualification decision is available. Confirmation, bootstrap qualification, a new V15 CSV and a fresh V15 submission notebook were not performed. Byte-identical scored V14 CSV/notebook are preserved as KEEP_V14 files.

Require positive gains on both development assignments and mean relative improvement >=0.20% against V14. One finalist is frozen before confirmation. Confirmation requires all three random gains positive, mean relative gain >=0.20%, a positive lower paired-row 95% gain bound (5,000 draws, seed 8319), and city/stratified deterioration <=1%. Labels have been historically examined; intervals are conditional robustness evidence, not an untouched test.

## Reproduction and compute

Conservative elapsed charge at packaging: 8.5880 hours under the eight-hour cap. Free T4 only; no portal upload.
V7 uses the independently verified original-Mac fitted state; fresh Linux V7 retraining is not claimed. Qualified standalone notebooks fit every required TabPFN component afresh. Expected predictions are only verification witnesses after inference.
Fresh same-layout and CSV agreement require INR 0.01. Permutation, split-batch and individual probes permit at most INR 10 per prediction and INR 1 pilot RMSE change. No rounding is used to pass checks.

Original inputs, fold witnesses, source/environment/checkpoint hashes and all measured failures accompany this report. At most one manual V15 portal trial is recommended after complete qualification and reproduction. Keep V14 until an actual lower portal RMSE is returned. The one-time coding upload remains unconfirmed.

Sources: Prior Labs TabPFN 9.2.0, official datetime/temperature documentation; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. See RESEARCH_SOURCES.md.

## Incomplete execution

This run stopped before every required stage completed. See execution_summary.json, stage_status.json and preserved run.log. No unmeasured comparison is claimed.

## Saved evidence verification

interruption_review.json independently verifies all four saved pilot identities/checks, zero completed accuracy folds, unchanged numerical sources and inputs, and all 439 preserved V14 files. Both KEEP_V14 files are byte-identical to the scored originals. A subsequent account check reported zero active assignments, zero hourly usage and zero compute-unit balance. No paid resources were purchased. No budget was reset; continuation requires a new explicit compute allowance.
