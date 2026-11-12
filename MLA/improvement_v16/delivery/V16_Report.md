# V16 measured report

**Retain V14: no new V16 submission is recommended.**

Best observed portal RMSE: V14, 11,736,110.7877709. V16 has no portal score. Local validation is not a portal result.

## Fixed experiment

Primary TM32-PLE: 32-member TabM, three 256-wide blocks, dropout 0.1, 32-bin/16-dimensional PLE, raw-target MSE, train-only preprocessing and inner epoch selection followed by all-outer-training refitting. MITRA2-FT is admitted only if its documented complete recipe and supported fitted-model reuse pass engineering.
Formula: (1-w)*V14+w*M, w in [0.10,0.25,0.50,0.75,1.00]. All 3,493 original labels remain in evaluation.

## Engineering

```json
{
  "status": "PASS",
  "settings": {
    "TM32-PLE": {
      "status": "PASS",
      "fold_seconds": 11.034096364999982,
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
        "name": "TM32-PLE",
        "training_rows": 2794,
        "units": "raw INR millions",
        "all_original_training_targets_retained": true,
        "members": 32,
        "inner_training_rows": 2374,
        "inner_validation_rows": 420,
        "inner_seed": 9091,
        "inner_policy": "random-15-percent",
        "inner_training_indices_sha256": "f4307178e65de4b077607b67933be8ddfc50c9cad5b124752311dad690bb8c98",
        "inner_validation_indices_sha256": "199ff1cb161f5dc00504df4b848fd36bf75f2945ec4c565d4c822184642cb809",
        "selected_epochs": 13,
        "inner_epochs_run": 33,
        "inner_history": [
          {
            "epoch": 1,
            "inner_rmse_INR": 17733180.950101532
          },
          {
            "epoch": 2,
            "inner_rmse_INR": 17202817.43176275
          },
          {
            "epoch": 3,
            "inner_rmse_INR": 17023737.538159665
          },
          {
            "epoch": 4,
            "inner_rmse_INR": 16735004.362212138
          },
          {
            "epoch": 5,
            "inner_rmse_INR": 16802864.038271267
          },
          {
            "epoch": 6,
            "inner_rmse_INR": 16645190.001604043
          },
          {
            "epoch": 7,
            "inner_rmse_INR": 16678180.907976637
          },
          {
            "epoch": 8,
            "inner_rmse_INR": 16392740.966800416
          },
          {
            "epoch": 9,
            "inner_rmse_INR": 16299408.60090266
          },
          {
            "epoch": 10,
            "inner_rmse_INR": 16299298.871090414
          },
          {
            "epoch": 11,
            "inner_rmse_INR": 16373243.190499995
          },
          {
            "epoch": 12,
            "inner_rmse_INR": 16302041.852919048
          },
          {
            "epoch": 13,
            "inner_rmse_INR": 16257626.601942973
          },
          {
            "epoch": 14,
            "inner_rmse_INR": 16306307.935638146
          },
          {
            "epoch": 15,
            "inner_rmse_INR": 16652098.690357588
          },
          {
            "epoch": 16,
            "inner_rmse_INR": 16948447.745920356
          },
          {
            "epoch": 17,
            "inner_rmse_INR": 16602848.894175982
          },
          {
            "epoch": 18,
            "inner_rmse_INR": 16498034.764773155
          },
          {
            "epoch": 19,
            "inner_rmse_INR": 16517836.714316547
          },
          {
            "epoch": 20,
            "inner_rmse_INR": 16556968.408765066
          },
          {
            "epoch": 21,
            "inner_rmse_INR": 16431361.913963713
          },
          {
            "epoch": 22,
            "inner_rmse_INR": 16447602.780103995
          },
          {
            "epoch": 23,
            "inner_rmse_INR": 16434646.715534702
          },
          {
            "epoch": 24,
            "inner_rmse_INR": 16608170.003986184
          },
          {
            "epoch": 25,
            "inner_rmse_INR": 16427632.340499569
          },
          {
            "epoch": 26,
            "inner_rmse_INR": 16471997.755121052
          },
          {
            "epoch": 27,
            "inner_rmse_INR": 16418979.47633712
          },
          {
            "epoch": 28,
            "inner_rmse_INR": 16423594.867881611
          },
          {
            "epoch": 29,
            "inner_rmse_INR": 16424290.109128144
          },
          {
            "epoch": 30,
            "inner_rmse_INR": 16455739.629159963
          },
          {
            "epoch": 31,
            "inner_rmse_INR": 16393891.538352171
          },
          {
            "epoch": 32,
            "inner_rmse_INR": 16523766.59437185
          },
          {
            "epoch": 33,
            "inner_rmse_INR": 16530526.959404781
          }
        ],
        "outer_validation_used_for_epoch_selection": false,
        "full_subset_refitted": true,
        "numeric_input_dimension": 41,
        "constant_numeric_inputs_dropped": 23,
        "target_mean_millions": 30.716535433070867,
        "target_scale_millions": 21.728727871451504,
        "fit_seconds": 10.935503857999947,
        "prediction_chunk": 500,
        "seed": 10242
      },
      "peak_gpu_allocated_bytes": 215374848,
      "training_rows": 2794,
      "validation_rows": 699,
      "engineering_seconds": 17.556519890000004
    },
    "MITRA2-FT": {
      "status": "engineering_excluded",
      "stage": "supported-fitted-model-reuse",
      "reason": "Official MitraFinetune.predict launches run_view for each query. run_view trains inside a TemporaryDirectory and returns arrays/distributions, then removes the fitted directory. The public API has no fitted-child reuse/load/predict interface. The predeclared persistent-inference gate cannot pass without changing the official adapter.",
      "repository": "autogluon/mitra-finetune",
      "revision": "b4701e8148dc33b00ed15d7086ff59816957cde4",
      "public_methods": [
        "fit",
        "predict",
        "predict_distribution",
        "predict_proba"
      ],
      "source_identity": {
        "repository": "autogluon/mitra-finetune",
        "revision": "b4701e8148dc33b00ed15d7086ff59816957cde4",
        "files": {
          "src/mitra_finetune/api.py": {
            "local_file": "improvement_v16/engineering_sources/mitra/api.py",
            "sha256": "c92c4fb0029329383a336168d3ce2c8be45186eed3f650dc59100d7cd2d3128c"
          },
          "src/mitra_finetune/runner.py": {
            "local_file": "improvement_v16/engineering_sources/mitra/runner.py",
            "sha256": "fbf912da747348a549ad33a2f33fd75d5d0a1982eccc863bc616345d1d0d761e"
          },
          "README.md": {
            "local_file": "improvement_v16/engineering_sources/mitra/README.md",
            "sha256": "1856dffa1f3107169f685e0596e231c17b7506c2d3e50a3e3f79f25be74c01e8"
          },
          "pyproject.toml": {
            "local_file": "improvement_v16/engineering_sources/mitra/pyproject.toml",
            "sha256": "1052134fe4ef95fab6bd061f3ce232f7a376bc026d3e85d803480048d98ed674"
          }
        }
      },
      "api_predict_line": 376,
      "run_view_line": 1116,
      "full_recipe_executed": false,
      "checkpoint_downloaded": false,
      "accuracy_tested": false,
      "selection_uses_validation_rmse": false,
      "budget_changed": false,
      "substitute_model_added": false
    }
  },
  "screen_settings": [
    "TM32-PLE"
  ],
  "engineering_feasible_settings": [
    "TM32-PLE"
  ],
  "budget_exclusions": {},
  "screen_forecast_seconds": 143.44325274499977,
  "confirmation_forecast_seconds": 358.6081318624994,
  "full_data_and_fresh_notebook_forecast_seconds": 1179.2035857912942,
  "export_forecast_basis": "Two full-data 512-epoch upper workload equivalents plus 600 seconds for freshly fitted controls and reporting; 30% contingency",
  "forecast_fits_budget": true,
  "selection_uses_validation_rmse": false,
  "sources": {
    "core.py": "21b3d4084b09d46618c84b09d5464635ea821a68a932e8f4db1cee2cc76a5bc3",
    "features.py": "2a14b0be1cf13e25f3d07e3b6ffee324d719a096aeaab3a2a60982b1cc822d78",
    "modeling.py": "f4ac06b064424e6059f09fcac6687e7850cb25945a6c3890b5b9157863f615b4",
    "workflow.py": "1bd9749a0fa7fd8ed13b88b372981ab20b66219b67b049ab1a34569ad39990e4",
    "verify.py": "14ebc89118bee90d66e81bf080da1b8bc0a26af641f7a82e246f78afcd2249af",
    "delivery.py": "4d2b8ffe5cad5a7ad7c2fe6229bb26b592048e85bcb797687b2c4577e58d9a92",
    "fresh_worker.py": "32ba32d75bfa8614a7bd88e89723b86f3fd06b926bb82b6451a564790e8c6aa6",
    "baseline.py": "0513655049b5d7c0e4e57b3339596e718779cd7bcbcc49bafd76e14e5bd9dad1",
    "legacy_models.py": "031c25ba5a60cf3e6e4c8247553ee951ff71742c846eb103246140929f6cdfba",
    "legacy_reference.py": "1a24f9f116170ea54bd3242264d5cd2a150c3429e039a50cb11c3112e52247c1",
    "frozen_v9_modeling.py": "c4d948bcf9e1aab5e667283eb82889af44bc57147a0c2a234af6322044b55487",
    "v5_contract.json": "3446412f01d536a31123142b984392216efd29094965500251386eac1814ab69",
    "city_map.json": "539387a0a9f5e143228714dde7679c2848a019a1cbbaa87129b0d4e4c092f7f9",
    "reference_manifest.json": "82417467e331d6e470fd1058c4ea4edb5a47f98021efc822176bd9a04b99bbd5",
    "baseline_registry.json": "53e7ddb7fbfbe08162aa97fac8d5ad9dea4981aab3aa942b3e98ebdb2a3060b7",
    "frozen_pfn.py": "ff4d383d269ffdc1b0021f7d0cac4e19240be130f2544bcc80dc41916a5262b5",
    "tabm_model.py": "9f616de239f0204424b68d35e76a85ea6eed4743ca8355a40ff3babd7259ebdf",
    "mitra_pilot.py": "b6bce4be92d909e103f6779dc8acd8e8873d8887511fe5708ba8e12789559967"
  },
  "inputs": {
    "Train_dataset.csv": "ec40ebda74ba0bfbfe1af4417a4266a70eadf6e595500e99e5435e7bdc7ee51f",
    "Test_dataset.csv": "ee4137566adb1c0ce356a8c370e7b8a22f681bde492068efe11346d8d44b4f4d",
    "Data_Dictionary.csv": "2d86c00e27448593564a161197b6639ee1eb90c01fa523b67e7a09d2ef405c5e",
    "Sample_Solution.csv": "9519ba4d51eb2d59b8a71b054a62e1f4d48b72cf92930c31f5e5393a2cd135fb"
  },
  "new_runtime": {
    "tabm": "0.0.3",
    "rtdl_num_embeddings": "0.0.12"
  },
  "model_contract_sha256": "85ad53fa889db58002c3a160a9d14ba3dc8f8c854a231258fb244eaf606037c5"
}
```

## Development

| Setting | Weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|---:|---:|
| TM32-PLE | 10% | 19,645,892.93 | 19,640,533.62 | 3,140.62 | 9,371.83 | 0.03184% |
| TM32-PLE | 25% | 19,652,832.45 | 19,643,192.47 | -3,798.89 | 6,712.98 | 0.00741% |
| TM32-PLE | 50% | 19,695,411.33 | 19,692,143.61 | -46,377.77 | -42,238.16 | -0.22549% |
| TM32-PLE | 75% | 19,776,519.71 | 19,796,345.94 | -127,486.15 | -146,440.48 | -0.69703% |
| TM32-PLE | 100% | 19,895,686.38 | 19,954,933.93 | -246,652.83 | -305,028.47 | -1.40380% |

## Decision and validation

Retain V14: no development candidate qualified.
Confirmation, bootstrap qualification, a new CSV and a fresh V16 submission notebook were skipped because no development candidate qualified. Byte-identical scored V14 files are preserved under KEEP_V14 names. Engineering exclusions are not accuracy failures.

Require positive gains on both development assignments and mean gain >=0.20% versus V14. Freeze one finalist before confirmation. Require all three random confirmation gains positive, mean relative gain >=0.20%, positive lower paired-row 95% gain bound (5,000 draws), and city/stratified deterioration <=1%. These are historically examined labels and conditional robustness checks.

## Reproduction and compute

Elapsed charge at packaging: 0.2533 hours under the six-hour cap, including setup and interruptions. Free T4 only; no portal upload.
Same-layout fresh reproduction and CSV agreement must be below INR 0.01. Permutation/batch/individual probes permit INR 10 prediction difference and INR 1 pilot RMSE change. No output rounding is used.
The V7 portion retains the verified original-Mac fitted-state boundary. Qualified notebooks freshly fit the required TabPFN controls and new component. Expected predictions are verification witnesses after inference.
At most one manual portal trial is recommended after every gate passes. Keep the lowest observed submission and leave the one-time coding submission pending final model choice.

Sources: official yandex-research/tabm and rtdl-num-embeddings implementations; autogluon/mitra-regressor-2 and autogluon/mitra-finetune; Prior Labs TabPFN. Exact sources, versions, hashes, failures and controls accompany the delivery.
