# V17 measured report

**Retain V14: no new V17 submission is recommended.**

Best observed portal RMSE: V14, 11,736,110.7877709. V17 has no portal score. Local validation is not a portal result.

## Fixed experiment

XIA32: base Xiaomi-TabLDM, 32 estimators. NOR15: Nori 1.5, four-member train-only recipe. P35FT512: one-estimator TabPFN-3.5 fine-tuning with 512-row context/query chunks and nested epoch selection/refit. Engineering admission is frozen before development scores.
Formula: (1-w)*V14+w*M, w in [0.10,0.25,0.50,0.75,1.00]. All 3,493 original labels remain in evaluation.

## Engineering

```json
{
  "status": "PASS",
  "settings": {
    "XIA32": {
      "status": "engineering_excluded",
      "error_type": "AssertionError",
      "reason": "permutation_difference_INR: INR 2290.7257080078125",
      "traceback": "Traceback (most recent call last):\n  File \"/content/mla_v17/pilot_worker.py\", line 31, in main\n    checks=verify.inference_checks(lambda raw:modeling.predict(fitted,raw),e.train.iloc[b],e.y[b])\n  File \"/content/mla_v17/verify.py\", line 137, in inference_checks\n    assert checks[name] <= 10., f'{name}: INR {checks[name]}'\n           ^^^^^^^^^^^^^^^^^^^\nAssertionError: permutation_difference_INR: INR 2290.7257080078125\n",
      "engineering_seconds": 489.56850253700003,
      "accuracy_tested": false
    },
    "NOR15": {
      "status": "PASS",
      "fold_seconds": 41.20470889699993,
      "full_fit_predict_seconds": 20.641729450000184,
      "behavior": {
        "permutation_difference_INR": 2.5902662128210068,
        "label_id_difference_INR": 0.0,
        "batch_difference_INR": 2.5902662128210068,
        "individual_difference_INR": 0.3237832747399807,
        "rmse_difference_INR": {
          "permutation": 0.0018624961376190186,
          "batch": 0.0003614462912082672,
          "individual": 0.04142097756266594
        },
        "other_query_rows_changed_difference_INR": 0.0,
        "rows_checked": 699,
        "individual_probes": 12,
        "missing_unseen_inputs": true,
        "behavior_tolerance_INR": 10.0,
        "rmse_tolerance_INR": 1.0
      },
      "fresh_repeat_difference_INR": 0.0,
      "save_reload_difference_INR": 0.0,
      "target_metadata": {
        "name": "NOR15",
        "training_rows": 2794,
        "units": "raw INR millions",
        "all_original_training_targets_retained": true,
        "seed": 10242,
        "training_target_sha256": "63998c05f2bd6fe27bf536d3f74ed91f5542c3bcef6e4bbc3ed28bc9492ffa38",
        "categorical_columns": [
          "Cuisine",
          "City",
          "Restaurant Location",
          "Endoresed By",
          "Restaurant Type",
          "Restaurant Theme",
          "Cuisine 1",
          "Cuisine 2"
        ],
        "features": 40,
        "actual_estimators": 4,
        "precision": "float32",
        "internal_selection": false,
        "public_interface": "NoriPredictor",
        "train_only_config_sha256": "64ad0d97763387b1373f98dbfb00225bd7aa1312b57291a871eb607f056832ec",
        "checkpoint_sha256": "b1313b45225066f74a78e99d5cd56dbd0015bdbeadf948252cf82d2a962766a2",
        "fit_seconds": 6.125843712999995
      },
      "peak_gpu_allocated_bytes": 3821158400,
      "training_rows": 2794,
      "validation_rows": 699,
      "full_training_rows": 3493,
      "selection_uses_validation_rmse": false,
      "engineering_seconds": 446.63141678399995
    },
    "P35FT512": {
      "status": "PASS",
      "fold_seconds": 510.89555311714224,
      "full_fit_predict_seconds": 605.4332475900001,
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
        "other_query_rows_changed_difference_INR": 0.0,
        "rows_checked": 699,
        "individual_probes": 12,
        "missing_unseen_inputs": true,
        "behavior_tolerance_INR": 10.0,
        "rmse_tolerance_INR": 1.0
      },
      "fresh_repeat_difference_INR": 0.0,
      "save_reload_difference_INR": 0.0,
      "target_metadata": {
        "name": "P35FT512",
        "training_rows": 2794,
        "units": "raw INR millions",
        "all_original_training_targets_retained": true,
        "seed": 10242,
        "training_target_sha256": "63998c05f2bd6fe27bf536d3f74ed91f5542c3bcef6e4bbc3ed28bc9492ffa38",
        "categorical_columns": [
          "Cuisine",
          "City",
          "Restaurant Location",
          "Endoresed By",
          "Restaurant Type",
          "Restaurant Theme",
          "Cuisine 1",
          "Cuisine 2"
        ],
        "features": 40,
        "matched_original_architectures": [
          "tabpfn.architectures.tabpfn_v3_5.TabPFNV3p5"
        ],
        "installed_finetune_source_sha256": "315413cd044720d1edc9962d6697ed55cea90f3e7181d1a2a1c41ac8ccde18fa",
        "inner_training_rows": 2374,
        "inner_validation_rows": 420,
        "inner_seed": 9091,
        "inner_train_sha256": "7f75c6d798f5f3e1aec1348dcd7f28ed151479de37588d9ef49616d76d44b982",
        "inner_validation_sha256": "78b66a263f116bce32eb4da28de7e21795134e6708d165edc8847f8b66befade",
        "inner_policy": "random-15-percent",
        "selected_epochs": 3,
        "inner_epochs_run": 11,
        "inner_history": [
          {
            "epoch": 0,
            "mse_millions_squared": 257.5474534175994
          },
          {
            "epoch": 1,
            "mse_millions_squared": 257.6810607380362
          },
          {
            "epoch": 2,
            "mse_millions_squared": 257.70148929933873
          },
          {
            "epoch": 3,
            "mse_millions_squared": 257.27076274186203
          },
          {
            "epoch": 4,
            "mse_millions_squared": 262.266338559799
          },
          {
            "epoch": 5,
            "mse_millions_squared": 294.52943309133315
          },
          {
            "epoch": 6,
            "mse_millions_squared": 305.45153409750327
          },
          {
            "epoch": 7,
            "mse_millions_squared": 310.19992460412044
          },
          {
            "epoch": 8,
            "mse_millions_squared": 281.1857909998409
          },
          {
            "epoch": 9,
            "mse_millions_squared": 284.5577516516772
          },
          {
            "epoch": 10,
            "mse_millions_squared": 281.46657196588126
          },
          {
            "epoch": 11,
            "mse_millions_squared": 281.69464363592556
          }
        ],
        "outer_validation_used_for_epoch_selection": false,
        "precision": "CUDA-AMP-training/float32-inference",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "matched_unadapted_inner_mse_millions_squared": 257.5474534175994,
        "full_subset_refitted": true,
        "actual_estimators": 1,
        "fit_seconds": 111.03168062600002
      },
      "peak_gpu_allocated_bytes": 5208716288,
      "training_rows": 2794,
      "validation_rows": 699,
      "full_training_rows": 3493,
      "selection_uses_validation_rmse": false,
      "engineering_seconds": 369.654790136,
      "forecast_basis": "Conservative maximum 30 selection plus 30 refit epochs, scaled from complete observed pilot."
    }
  },
  "screen_settings": [
    "NOR15"
  ],
  "engineering_feasible_settings": [
    "NOR15",
    "P35FT512"
  ],
  "budget_exclusions": {
    "P35FT512": "Full development, slowest-finalist confirmation and fresh-delivery workload forecast exceeds accepted budget"
  },
  "screen_forecast_seconds": 535.6612156609991,
  "confirmation_forecast_seconds": 1339.1530391524977,
  "full_data_and_fresh_notebook_forecast_seconds": 1613.6684965700006,
  "forecast_fits_budget": true,
  "selection_uses_validation_rmse": false,
  "sources": {
    "core.py": "5312eada9c348afa96e92185145fd91d19ad4500a9b4913ee2efcc229e94cb8a",
    "features.py": "4173ca9923818419245ae723a8487be4d12385a194e16617d43e25663f86133c",
    "modeling.py": "ed7a0703ce2e9573cf99f3f0a5a5b4a62da4db4ea5790450b3d7bcfb55aa4b0b",
    "workflow.py": "795a6fa926ce9167d371cc9ec26a0b75a2eaf5412279a81d00b4777223d9a9de",
    "verify.py": "8fa80eda1db87cbf7220854462670717514d780af202e57becf016ec3e2773ca",
    "delivery.py": "be619d554a6ca2a0d38246ca7d848d3e228cd9da8849f57cefa330ea55923079",
    "fresh_worker.py": "c76bde02db74b8b8d9ecc17dc9b20fd9a61e1eaab0ab5c6325a8f1ea6c5f4936",
    "baseline.py": "0513655049b5d7c0e4e57b3339596e718779cd7bcbcc49bafd76e14e5bd9dad1",
    "legacy_models.py": "031c25ba5a60cf3e6e4c8247553ee951ff71742c846eb103246140929f6cdfba",
    "legacy_reference.py": "1a24f9f116170ea54bd3242264d5cd2a150c3429e039a50cb11c3112e52247c1",
    "frozen_v9_modeling.py": "c4d948bcf9e1aab5e667283eb82889af44bc57147a0c2a234af6322044b55487",
    "v5_contract.json": "3446412f01d536a31123142b984392216efd29094965500251386eac1814ab69",
    "city_map.json": "539387a0a9f5e143228714dde7679c2848a019a1cbbaa87129b0d4e4c092f7f9",
    "reference_manifest.json": "82417467e331d6e470fd1058c4ea4edb5a47f98021efc822176bd9a04b99bbd5",
    "baseline_registry.json": "53e7ddb7fbfbe08162aa97fac8d5ad9dea4981aab3aa942b3e98ebdb2a3060b7",
    "frozen_pfn.py": "ff4d383d269ffdc1b0021f7d0cac4e19240be130f2544bcc80dc41916a5262b5",
    "new_models.py": "60f7c9e96987a67fe3b7bc1a5fc4fb3694f3874b7a27aa50cadfc4958e8feb40",
    "pilot_worker.py": "7d434c6061a67825ce43470ffe4a26e000b8ea15a09d21398e847ccfe5c35654",
    "checkpoint_registry.json": "a32a93bb96478383073c3483e77454fb490cc06ab2e35002041523ef62decc69",
    "nori_inference.json": "64ad0d97763387b1373f98dbfb00225bd7aa1312b57291a871eb607f056832ec",
    "source_registry.json": "2ee52e52f9bed2600e412d0f016d0e4e87b9baea6932f46c2b558a1a0ff5ac61"
  },
  "inputs": {
    "Train_dataset.csv": "ec40ebda74ba0bfbfe1af4417a4266a70eadf6e595500e99e5435e7bdc7ee51f",
    "Test_dataset.csv": "ee4137566adb1c0ce356a8c370e7b8a22f681bde492068efe11346d8d44b4f4d",
    "Data_Dictionary.csv": "2d86c00e27448593564a161197b6639ee1eb90c01fa523b67e7a09d2ef405c5e",
    "Sample_Solution.csv": "9519ba4d51eb2d59b8a71b054a62e1f4d48b72cf92930c31f5e5393a2cd135fb"
  },
  "new_runtime": {
    "einops": "0.8.2",
    "numba": "0.68.1",
    "llvmlite": "0.50.0",
    "kditransform": "1.2.0",
    "pyvinecopulib": "1.0.1",
    "synthefy": "7.4.0",
    "pyyaml": "6.0.3",
    "pydantic": "2.14.1",
    "setuptools": "80.10.2"
  },
  "model_contract_sha256": "33f95b5eed3ce8ee9b365594ea010b2cb04474fadf09c4118a976c93e27a2cf2"
}
```

## Development

| Setting | Weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|---:|---:|
| NOR15 | 10% | 19,641,071.15 | 19,643,452.74 | 7,962.40 | 6,452.71 | 0.03668% |
| NOR15 | 25% | 19,632,364.35 | 19,636,949.18 | 16,669.20 | 12,956.27 | 0.07539% |
| NOR15 | 50% | 19,626,495.84 | 19,634,585.92 | 22,537.72 | 15,319.53 | 0.09633% |
| NOR15 | 75% | 19,631,437.70 | 19,642,819.50 | 17,595.85 | 7,085.96 | 0.06281% |
| NOR15 | 100% | 19,647,181.79 | 19,661,636.59 | 1,851.77 | -11,731.14 | -0.02514% |

## Decision and validation

Retain V14: no development candidate qualified.
Confirmation, bootstrap qualification, a new CSV and a fresh V17 submission notebook were skipped because no development candidate qualified. Byte-identical scored V14 files are preserved under KEEP_V14 names. Engineering exclusions are not accuracy failures.

Require positive gains on both development assignments and mean gain >=0.20% versus V14. Freeze one finalist before confirmation. Require all three random confirmation gains positive, mean relative gain >=0.20%, positive lower paired-row 95% gain bound (5,000 draws), and city/stratified deterioration <=1%. These are historically examined labels and conditional robustness checks.

## Reproduction and compute

Elapsed charge at packaging: 0.9009 hours under the six-hour cap, including setup and interruptions. Free T4 only; no portal upload.
Same-layout fresh reproduction and CSV agreement must be below INR 0.01. Permutation/batch/individual probes permit INR 10 prediction difference and INR 1 pilot RMSE change. No output rounding is used.
The V7 portion retains the verified original-Mac fitted-state boundary. Qualified notebooks freshly fit the required TabPFN controls and new component. Expected predictions are verification witnesses after inference.
At most one manual portal trial is recommended after every gate passes. Keep the lowest observed submission and leave the one-time coding submission pending final model choice.

Sources: official Xiaomi-TabLDM, Synthefy Nori and Prior Labs TabPFN repositories. Exact sources, versions, hashes, failures and controls accompany the delivery.
