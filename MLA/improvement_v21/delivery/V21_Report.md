# V21 measured report

**Retain V14: no new V21 submission is recommended.**

Best observed portal RMSE: V14, 11,736,110.7877709. V21 has not been uploaded. Local validation is not a portal score.

This is a separately authorized two-representation feature experiment. The six-hour operational ledger includes implementation, setup, failures and interruptions; reserve 1.5 hours for delivery. Earlier version ledgers remain unchanged.

## Fixed experiment

Two settings: S (56 features) and C+S+G (66 features), reusing reviewed V10 definitions with the exact V14 TabPFN-3.5 checkpoint, eight estimators, seed 10242 and original labels in INR millions. Group maps and categorical vocabularies fit on training rows only. Checkpoint temperature/calibration defaults are unchanged. V10 tested these features with v2; performance with 3.5 was unproven.

Formula: .25*V9 + .75*((1-w)*original_P35 + w*new_component), w in [.25,.5,1]. Six comparisons maximum. Baseline is scored V14.

## Engineering pilots

```json
{
  "status": "PASS",
  "settings": {
    "S": {
      "status": "PASS",
      "fold_seconds": 25.35211871299998,
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
        "name": "S",
        "training_rows": 2794,
        "units": "raw INR millions",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "model_class": "tabpfn.regressor.TabPFNRegressor",
        "view": "S",
        "input_features": 56,
        "temperature": "checkpoint default",
        "feature_state_sha256": "1b1f181a6265017147f6bc5b996d997653324c165e0809ceabf11db0d2e37b44",
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
      "full_data_seconds": 18.487362490000123,
      "full_data_training_rows": 3493,
      "full_data_query_rows": 500,
      "reload_difference_INR": 0.0,
      "peak_gpu_allocated_bytes": 9508628480,
      "training_rows": 2794,
      "validation_rows": 699,
      "engineering_seconds": 308.50722249699993
    },
    "C+S+G": {
      "status": "PASS",
      "fold_seconds": 28.234828268,
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
        "name": "C+S+G",
        "training_rows": 2794,
        "units": "raw INR millions",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "model_class": "tabpfn.regressor.TabPFNRegressor",
        "view": "C+S+G",
        "input_features": 66,
        "temperature": "checkpoint default",
        "feature_state_sha256": "eaaff75ee34aea51b240f6a3d498fe5fd8f01562024482694c84c8899f478164",
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
      "full_data_seconds": 19.42131909999989,
      "full_data_training_rows": 3493,
      "full_data_query_rows": 500,
      "reload_difference_INR": 0.0,
      "peak_gpu_allocated_bytes": 9367408128,
      "training_rows": 2794,
      "validation_rows": 699,
      "engineering_seconds": 344.5265451350001
    }
  },
  "screen_settings": [
    "S",
    "C+S+G"
  ],
  "engineering_feasible_settings": [
    "S",
    "C+S+G"
  ],
  "budget_exclusions": {},
  "screen_forecast_seconds": 696.6303107529998,
  "confirmation_forecast_seconds": 917.63191871,
  "delivery_forecast_seconds": 541.4862889799991,
  "forecast_fits_budget": true,
  "selection_uses_validation_rmse": false,
  "sources": {
    "core.py": "92f2fc6fd733253245d636054b95fd94970a8ffe34cfe70f24fbbe7d21b9895a",
    "features.py": "786a081eaef3de0ad2df9a96748399306182a8210870216445dd80dd610e21a9",
    "modeling.py": "bfb1a0f610f3dc3df416121a8a384d96044cefc3b5cca99cce447761a6e21149",
    "workflow.py": "b260062000707e38f2806cdb45391875537c48eb0768d40d530a5ebfccb95909",
    "verify.py": "f842179ced4b4f33aa0ed6b9130c125079f7476f69aff773121f2f61ee23a3b8",
    "delivery.py": "74db6188d2442d9939c448cfc6815c7f62d72778cfccc8730c6ba19762143852",
    "fresh_worker.py": "3837f6189439918919e2698b4ffca84e504e307897c4186117a316e401f1461c",
    "baseline.py": "0513655049b5d7c0e4e57b3339596e718779cd7bcbcc49bafd76e14e5bd9dad1",
    "legacy_models.py": "031c25ba5a60cf3e6e4c8247553ee951ff71742c846eb103246140929f6cdfba",
    "legacy_reference.py": "1a24f9f116170ea54bd3242264d5cd2a150c3429e039a50cb11c3112e52247c1",
    "frozen_v9_modeling.py": "c4d948bcf9e1aab5e667283eb82889af44bc57147a0c2a234af6322044b55487",
    "v5_contract.json": "3446412f01d536a31123142b984392216efd29094965500251386eac1814ab69",
    "city_map.json": "539387a0a9f5e143228714dde7679c2848a019a1cbbaa87129b0d4e4c092f7f9",
    "reference_manifest.json": "82417467e331d6e470fd1058c4ea4edb5a47f98021efc822176bd9a04b99bbd5",
    "baseline_registry.json": "53e7ddb7fbfbe08162aa97fac8d5ad9dea4981aab3aa942b3e98ebdb2a3060b7",
    "PLAN.md": "1251b7e787b0aa096d80f1156da5e043bf0768bbc2cc151efb1ede3c97d04d83",
    "authorization.json": "863ef36aea6ae7743f0387edaba0e2d33e2789d88204414e00bad4c27da1763b",
    "test_v21.py": "daa4ccceefeabe50da44e451b4adce4f029868beaa6a2649d97d047b18c3b12a",
    "verify_results.py": "41dd0e06c44b58ec04ce4dd8e8e8c7eb2ff8ce9b25c66aff584bb752ff5ba8d7"
  },
  "inputs": {
    "Train_dataset.csv": "ec40ebda74ba0bfbfe1af4417a4266a70eadf6e595500e99e5435e7bdc7ee51f",
    "Test_dataset.csv": "ee4137566adb1c0ce356a8c370e7b8a22f681bde492068efe11346d8d44b4f4d",
    "Data_Dictionary.csv": "2d86c00e27448593564a161197b6639ee1eb90c01fa523b67e7a09d2ef405c5e",
    "Sample_Solution.csv": "9519ba4d51eb2d59b8a71b054a62e1f4d48b72cf92930c31f5e5393a2cd135fb"
  },
  "model_contract_sha256": "bae95f41a0b561c6df81c536ad5bb4a87231428720ef1e500e8dd4519fab39a6"
}
```

## Development

| Setting | Replacement weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|---:|---:|
| S | 25% | 19,646,252.06 | 19,650,776.67 | 2,781.50 | -871.21 | 0.00486% |
| S | 50% | 19,646,011.78 | 19,654,235.81 | 3,021.78 | -4,330.36 | -0.00333% |
| S | 100% | 19,653,154.28 | 19,668,911.40 | -4,120.72 | -19,005.95 | -0.05885% |
| C+S+G | 25% | 19,653,293.81 | 19,654,110.13 | -4,260.25 | -4,204.68 | -0.02154% |
| C+S+G | 50% | 19,661,501.87 | 19,662,249.21 | -12,468.31 | -12,343.75 | -0.06314% |
| C+S+G | 100% | 19,689,739.30 | 19,690,308.67 | -40,705.74 | -40,403.22 | -0.20639% |

## Confirmation and decision

Retain V14: no development candidate qualified.
No development finalist qualified. Confirmation, bootstrap qualification, a new V21 CSV and a fresh V21 submission notebook were correctly skipped. Byte-identical scored V14 CSV/notebook are preserved as KEEP_V14 files.

Require positive gains on both development assignments and mean relative improvement >=0.20% against V14. One finalist is frozen before confirmation. Confirmation requires all three random gains positive, mean relative gain >=0.20%, a positive lower paired-row 95% gain bound (5,000 draws, seed 8319), and city/stratified deterioration <=1%. Labels have been historically examined; intervals are conditional robustness evidence, not an untouched test.

## Reproduction and compute

Conservative elapsed charge at packaging: 0.5801 hours under the six-hour cap. Free T4 only; no portal upload.
V7 uses the independently verified original-Mac fitted state; fresh Linux V7 retraining is not claimed. Qualified standalone notebooks fit every required TabPFN component afresh. Expected predictions are only verification witnesses after inference.
Fresh same-layout and CSV agreement require INR 0.01. Permutation, split-batch and individual probes permit at most INR 10 per prediction and INR 1 pilot RMSE change. No rounding is used to pass checks.

Original inputs, fold witnesses, source/environment/checkpoint hashes and all measured failures accompany this report. At most one manual V21 portal trial is recommended after complete qualification and reproduction. Keep V14 until an actual lower portal RMSE is returned. The one-time coding upload remains unconfirmed.

Sources: Prior Labs TabPFN 9.2.0, reviewed local V10 feature definitions; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. See RESEARCH_SOURCES.md.
