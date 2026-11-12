# V19 measured report

**Retain V14: no new V19 submission is recommended.**

Best observed portal RMSE: V14, 11,736,110.7877709. V19 has no portal score. Local validation is not a portal result.

## Fixed experiment

C8: eight-member Causilo. KS8/KM8: Small/Medium NVIDIA Kumo with eight members and a fixed quantile-grid mean. D8: eight-member TabDPT v1.3 using its non-Flash path. One equal mean of admitted core families is predeclared. Engineering admission is frozen before development scores.
Formula: (1-w)*V14+w*M, w in [0.10,0.25,0.50,0.75,1.00]. All 3,493 original labels remain in evaluation.

## Engineering

```json
{
  "status": "PASS",
  "settings": {
    "C8": {
      "status": "PASS",
      "fold_seconds": 18.014813181000022,
      "full_fit_predict_seconds": 9.694622115000016,
      "behavior": {
        "permutation_difference_INR": 0.647566556930542,
        "label_id_difference_INR": 0.0,
        "batch_difference_INR": 1.1332414597272873,
        "individual_difference_INR": 0.3237832710146904,
        "rmse_difference_INR": {
          "permutation": 0.003399420529603958,
          "batch": 0.0014055818319320679,
          "individual": 0.03881894238293171
        },
        "other_query_rows_changed_difference_INR": 0.0,
        "rows_checked": 699,
        "individual_probes": 12,
        "missing_unseen_inputs": true,
        "behavior_tolerance_INR": 10.0,
        "prediction_rtol": 1e-06,
        "rmse_tolerance_INR": 10.0
      },
      "fresh_repeat_difference_INR": 0.0,
      "save_reload_difference_INR": 0.0,
      "target_metadata": {
        "name": "C8",
        "training_rows": 2794,
        "units": "raw INR millions",
        "model_class": "causilo.estimators.CausiloRegressor",
        "checkpoint_sha256": "9a4197d7c1060d187c19b62d4d83c74ce986c1f3911825576eb58c2ae4f0057d",
        "checkpoint_revision": "94f2bd91db0737d4da59f347910662905ecb5a09",
        "actual_source_hashes": {
          "causilo.estimators": "74d35df8d35f681c3c458522a15d2bcd10e1f4aa35a53ae240d16d9686b3bbbf",
          "causilo.checkpoints": "1c2f1577e61ea6fdcd8f0593610862be4d539a99fa8a62812c8f4a253aaf36d4",
          "causilo.nn.layers.attention": "3aa33b9ea4f6d5caa2fb9e38233ed67d3d6fefdd7a44857c50d9017b91c853ea"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.13.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "causilo": "1.0.3",
            "tabdpt": "1.3.1",
            "numba": "0.68.0",
            "llvmlite": "0.50.0",
            "structured-data-models": "0.1.0.dev0",
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208
        },
        "input_features": 40,
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
        "all_outer_training_rows": true,
        "seed": 10242,
        "peak_gpu_allocated_bytes": 2477054976,
        "peak_gpu_reserved_bytes": 4309647360,
        "host_rss_bytes": 1721704448,
        "query_layout": "256 rows, padded with the fixed first training row; target and ID excluded"
      },
      "full_target_metadata": {
        "name": "C8",
        "training_rows": 3493,
        "units": "raw INR millions",
        "model_class": "causilo.estimators.CausiloRegressor",
        "checkpoint_sha256": "9a4197d7c1060d187c19b62d4d83c74ce986c1f3911825576eb58c2ae4f0057d",
        "checkpoint_revision": "94f2bd91db0737d4da59f347910662905ecb5a09",
        "actual_source_hashes": {
          "causilo.estimators": "74d35df8d35f681c3c458522a15d2bcd10e1f4aa35a53ae240d16d9686b3bbbf",
          "causilo.checkpoints": "1c2f1577e61ea6fdcd8f0593610862be4d539a99fa8a62812c8f4a253aaf36d4",
          "causilo.nn.layers.attention": "3aa33b9ea4f6d5caa2fb9e38233ed67d3d6fefdd7a44857c50d9017b91c853ea"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.13.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "causilo": "1.0.3",
            "tabdpt": "1.3.1",
            "numba": "0.68.0",
            "llvmlite": "0.50.0",
            "structured-data-models": "0.1.0.dev0",
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208
        },
        "input_features": 40,
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
        "all_outer_training_rows": true,
        "seed": 10242,
        "peak_gpu_allocated_bytes": 3003019776,
        "peak_gpu_reserved_bytes": 5265948672,
        "host_rss_bytes": 1613565952,
        "query_layout": "256 rows, padded with the fixed first training row; target and ID excluded"
      },
      "peak_gpu_allocated_bytes": 2477054976,
      "peak_gpu_reserved_bytes": 4309647360,
      "training_rows": 2794,
      "validation_rows": 699,
      "full_training_rows": 3493,
      "selection_uses_validation_rmse": false,
      "engineering_seconds": 97.03965899499997
    },
    "KS8": {
      "status": "engineering_excluded",
      "error_type": "RuntimeError",
      "reason": "AttributeError: module 'numba.cuda.types' has no attribute 'NPDatetime'",
      "traceback": "Traceback (most recent call last):\n  File \"/content/mla_v19/pilot_worker.py\", line 27, in main\n    fitted=modeling.fit(e.train.iloc[a],e.y[a],name,fold=0,scheme=42)\n  File \"/content/mla_v19/modeling.py\", line 42, in fit\n    fitted = new_models.fit(raw, target, name, fold=fold, scheme=scheme)\n  File \"/content/mla_v19/new_models.py\", line 128, in fit\n    metadata = remote.request('fit', {'raw': raw, 'target': np.asarray(target, float), 'name': name})\n  File \"/content/mla_v19/new_models.py\", line 94, in request\n    raise RuntimeError(status['error_type'] + ': ' + status['reason'])\nRuntimeError: AttributeError: module 'numba.cuda.types' has no attribute 'NPDatetime'\n",
      "engineering_seconds": 18.355456892999996,
      "accuracy_tested": false
    },
    "D8": {
      "status": "PASS",
      "fold_seconds": 20.604503259000012,
      "full_fit_predict_seconds": 14.271207941000057,
      "behavior": {
        "permutation_difference_INR": 7.62939453125,
        "label_id_difference_INR": 0.0,
        "batch_difference_INR": 7.62939453125,
        "individual_difference_INR": 15.2587890625,
        "rmse_difference_INR": {
          "permutation": 0.01396353542804718,
          "batch": 0.20910362154245377,
          "individual": 2.8792647439986467
        },
        "other_query_rows_changed_difference_INR": 0.0,
        "rows_checked": 699,
        "individual_probes": 12,
        "missing_unseen_inputs": true,
        "behavior_tolerance_INR": 10.0,
        "prediction_rtol": 1e-06,
        "rmse_tolerance_INR": 10.0
      },
      "fresh_repeat_difference_INR": 0.0,
      "save_reload_difference_INR": 0.0,
      "target_metadata": {
        "name": "D8",
        "training_rows": 2794,
        "units": "raw INR millions",
        "model_class": "tabdpt.regressor.TabDPTRegressor",
        "checkpoint_sha256": "97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb",
        "checkpoint_revision": "a5ca6e01c0fa09ec68c73e958e5199d1932abb3a",
        "actual_source_hashes": {
          "tabdpt.regressor": "03f420c110e19039dbbd912d80ecb36c407292cc96ce41de91d75d43992e9c2e",
          "tabdpt.estimator": "81a345826a5534d9e4a5916ab4dd11f815a17787703eb54a42b7c968a2992678",
          "tabdpt.model": "f7059d9040d46d883bc9f4df1456c77c369f9b3d9d59c3bf31c7e077af0fb381",
          "tabdpt.utils": "5b6eb9ad2f49f18e99d1a850209263924ff8a2941a0b976a57511408803237c7"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.13.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "causilo": "1.0.3",
            "tabdpt": "1.3.1",
            "numba": "0.68.0",
            "llvmlite": "0.50.0",
            "structured-data-models": "0.1.0.dev0",
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208
        },
        "input_features": 40,
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
        "all_outer_training_rows": true,
        "seed": 10242,
        "peak_gpu_allocated_bytes": 509666304,
        "peak_gpu_reserved_bytes": 564133888,
        "host_rss_bytes": 1667731456
      },
      "full_target_metadata": {
        "name": "D8",
        "training_rows": 3493,
        "units": "raw INR millions",
        "model_class": "tabdpt.regressor.TabDPTRegressor",
        "checkpoint_sha256": "97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb",
        "checkpoint_revision": "a5ca6e01c0fa09ec68c73e958e5199d1932abb3a",
        "actual_source_hashes": {
          "tabdpt.regressor": "03f420c110e19039dbbd912d80ecb36c407292cc96ce41de91d75d43992e9c2e",
          "tabdpt.estimator": "81a345826a5534d9e4a5916ab4dd11f815a17787703eb54a42b7c968a2992678",
          "tabdpt.model": "f7059d9040d46d883bc9f4df1456c77c369f9b3d9d59c3bf31c7e077af0fb381",
          "tabdpt.utils": "5b6eb9ad2f49f18e99d1a850209263924ff8a2941a0b976a57511408803237c7"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.13.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "causilo": "1.0.3",
            "tabdpt": "1.3.1",
            "numba": "0.68.0",
            "llvmlite": "0.50.0",
            "structured-data-models": "0.1.0.dev0",
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208
        },
        "input_features": 40,
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
        "all_outer_training_rows": true,
        "seed": 10242,
        "peak_gpu_allocated_bytes": 509666304,
        "peak_gpu_reserved_bytes": 580911104,
        "host_rss_bytes": 1622183936
      },
      "peak_gpu_allocated_bytes": 509666304,
      "peak_gpu_reserved_bytes": 564133888,
      "training_rows": 2794,
      "validation_rows": 699,
      "full_training_rows": 3493,
      "selection_uses_validation_rmse": false,
      "engineering_seconds": 158.35212486
    },
    "KM8": {
      "status": "engineering_excluded",
      "error_type": "RuntimeError",
      "reason": "AttributeError: module 'numba.cuda.types' has no attribute 'NPDatetime'",
      "traceback": "Traceback (most recent call last):\n  File \"/content/mla_v19/pilot_worker.py\", line 27, in main\n    fitted=modeling.fit(e.train.iloc[a],e.y[a],name,fold=0,scheme=42)\n  File \"/content/mla_v19/modeling.py\", line 42, in fit\n    fitted = new_models.fit(raw, target, name, fold=fold, scheme=scheme)\n  File \"/content/mla_v19/new_models.py\", line 128, in fit\n    metadata = remote.request('fit', {'raw': raw, 'target': np.asarray(target, float), 'name': name})\n  File \"/content/mla_v19/new_models.py\", line 94, in request\n    raise RuntimeError(status['error_type'] + ': ' + status['reason'])\nRuntimeError: AttributeError: module 'numba.cuda.types' has no attribute 'NPDatetime'\n",
      "engineering_seconds": 17.791377285999943,
      "accuracy_tested": false
    }
  },
  "screen_settings": [
    "C8",
    "D8"
  ],
  "routes": {
    "C8": {
      "C8": 1.0
    },
    "D8": {
      "D8": 1.0
    },
    "CORE_EQUAL": {
      "C8": 0.5,
      "D8": 0.5
    }
  },
  "engineering_feasible_settings": [
    "C8",
    "D8"
  ],
  "budget_exclusions": {},
  "screen_forecast_seconds": 502.05111372000044,
  "confirmation_forecast_seconds": 5897.880286720662,
  "full_data_and_fresh_notebook_forecast_seconds": 2402.3111581456,
  "forecast_fits_budget": true,
  "selection_uses_validation_rmse": false,
  "sources": {
    "core.py": "0f912c0ae24adf244291d31b6a8348a073eaf8086e23059aaf36d6c6a500463e",
    "features.py": "a033e459df5b9776f95a9ef683ccba174b1edaab7f63901c50cfa71e9bee8b99",
    "modeling.py": "e344038c0445ab87742dc4b00f07dd919b32a6a228bc8b74a87f823d3c429e3c",
    "workflow.py": "eee5ef74da8af9752f67519c7f9e72d3e4633acce4f6917952fe6712ded9ac3e",
    "verify.py": "650a4e5fa1e25060689a575f97305ac27153a7e5b4aa3ac2c64c08ff5ca598ff",
    "delivery.py": "df7b6a39f220e40faf1b377b0c57a0c00687cf3e1fdb76239960a9b4025188d5",
    "fresh_worker.py": "99141f44ed395f25c00f8241aca4c3eb260e7dacc80a1a3a811af88ecefc01ec",
    "baseline.py": "0513655049b5d7c0e4e57b3339596e718779cd7bcbcc49bafd76e14e5bd9dad1",
    "legacy_models.py": "031c25ba5a60cf3e6e4c8247553ee951ff71742c846eb103246140929f6cdfba",
    "legacy_reference.py": "1a24f9f116170ea54bd3242264d5cd2a150c3429e039a50cb11c3112e52247c1",
    "frozen_v9_modeling.py": "c4d948bcf9e1aab5e667283eb82889af44bc57147a0c2a234af6322044b55487",
    "v5_contract.json": "3446412f01d536a31123142b984392216efd29094965500251386eac1814ab69",
    "city_map.json": "539387a0a9f5e143228714dde7679c2848a019a1cbbaa87129b0d4e4c092f7f9",
    "reference_manifest.json": "82417467e331d6e470fd1058c4ea4edb5a47f98021efc822176bd9a04b99bbd5",
    "baseline_registry.json": "53e7ddb7fbfbe08162aa97fac8d5ad9dea4981aab3aa942b3e98ebdb2a3060b7",
    "frozen_pfn.py": "ff4d383d269ffdc1b0021f7d0cac4e19240be130f2544bcc80dc41916a5262b5",
    "new_models.py": "a060a2d5857267a2d188e2df52608927d738d313c72f59e3e952573448da9940",
    "candidate_models.py": "90848b6804c069347b852a8237fb5841c894af89dcf19f66cfb27324907e1f3d",
    "candidate_worker.py": "a70cb7f5406a194eacbd7e653a84c0c5f60480fa602799be31993ff8049c9e0d",
    "pilot_worker.py": "5f16c9e4902d97a91875c043b4e1e2cd118650d9cd016cd000181667a938e7c8",
    "checkpoint_registry.json": "d4989d804e9295c9a8f6087205b3de52e737a8aa3f16c06d87c59b3a63daeb3b",
    "source_registry.json": "1377f16dee1a9b08e8d2b3d5bb36713a79fa48f015af79baf1110ee5ccf69273",
    "protocol_proposed.json": "461311b69d0ed00c2d5a53cbcedfd694b55259e088fcddadb10b32251f1fa6a2",
    "mac_controls.py": "c6c01ea36a0a9f3a4c5a6e4c8110c4e07b83e5373c110e075f4a7844515208c3",
    "control_worker.py": "79c40d599b5aff4a3a120464723d9fd1075200195787d7e4e157827e10945fca",
    "setup_environment.py": "a0b904b90eb27e9dd104312f648a9b86a559a0093a2abb5ac93b7d0fe61d9066",
    "runtime_lock.json": "d3dd0fec6b990c17cfddd101ce5b2100d0c09f7412eb93546304a98aa628aec6"
  },
  "inputs": {
    "Train_dataset.csv": "ec40ebda74ba0bfbfe1af4417a4266a70eadf6e595500e99e5435e7bdc7ee51f",
    "Test_dataset.csv": "ee4137566adb1c0ce356a8c370e7b8a22f681bde492068efe11346d8d44b4f4d",
    "Data_Dictionary.csv": "2d86c00e27448593564a161197b6639ee1eb90c01fa523b67e7a09d2ef405c5e",
    "Sample_Solution.csv": "9519ba4d51eb2d59b8a71b054a62e1f4d48b72cf92930c31f5e5393a2cd135fb"
  },
  "model_contract_sha256": "033bfda1fed4909428a523b2659cc6dde73332ab7ff3f643a32cad79e8d3379f",
  "control_timings": {
    "BASE": {
      "status": "PASS",
      "seconds": 122.48086434699997,
      "maximum_existing_fold_difference_INR": 0.0
    },
    "P35_8": {
      "status": "PASS",
      "seconds": 34.566835114000014,
      "maximum_existing_fold_difference_INR": 0.0
    }
  },
  "original_mac_control_timing": {
    "status": "PASS",
    "seconds": 81.04217245800828,
    "maximum_difference_INR": 1.4901161193847656e-08
  }
}
```

## Development

| Setting | Weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|---:|---:|
| C8 | 10% | 19,647,318.09 | 19,647,276.46 | 1,715.47 | 2,628.99 | 0.01105% |
| C8 | 25% | 19,647,339.01 | 19,646,415.48 | 1,694.55 | 3,489.97 | 0.01319% |
| C8 | 50% | 19,654,290.59 | 19,653,200.46 | -5,257.04 | -3,295.01 | -0.02176% |
| C8 | 75% | 19,669,879.15 | 19,670,249.76 | -20,845.59 | -20,344.31 | -0.10481% |
| C8 | 100% | 19,694,084.16 | 19,697,536.73 | -45,050.61 | -47,631.28 | -0.23584% |
| D8 | 10% | 19,657,050.38 | 19,658,509.09 | -8,016.82 | -8,603.64 | -0.04229% |
| D8 | 25% | 19,677,489.83 | 19,679,540.87 | -28,456.28 | -29,635.42 | -0.14782% |
| D8 | 50% | 19,733,906.49 | 19,736,178.15 | -84,872.93 | -86,272.70 | -0.43550% |
| D8 | 75% | 19,818,044.72 | 19,819,585.81 | -169,011.17 | -169,680.36 | -0.86183% |
| D8 | 100% | 19,929,553.45 | 19,929,427.75 | -280,519.89 | -279,522.29 | -1.42508% |
| CORE_EQUAL | 10% | 19,651,602.43 | 19,652,322.58 | -2,568.87 | -2,417.12 | -0.01269% |
| CORE_EQUAL | 25% | 19,658,781.77 | 19,659,417.96 | -9,748.22 | -9,512.50 | -0.04901% |
| CORE_EQUAL | 50% | 19,679,604.48 | 19,680,483.37 | -30,570.92 | -30,577.92 | -0.15560% |
| CORE_EQUAL | 75% | 19,711,466.58 | 19,713,064.66 | -62,433.02 | -63,159.20 | -0.31958% |
| CORE_EQUAL | 100% | 19,754,314.66 | 19,757,104.85 | -105,281.10 | -107,199.39 | -0.54068% |

## Decision and validation

Retain V14: no development candidate qualified.
Confirmation, bootstrap qualification, a new CSV and a fresh V19 submission notebook were skipped because no development candidate qualified. Byte-identical scored V14 files are preserved under KEEP_V14 names. Engineering exclusions are not accuracy failures.

Require positive gains on both development assignments and mean gain >=0.20% versus V14. Freeze one finalist before confirmation. Require all three random confirmation gains positive, mean relative gain >=0.20%, positive lower paired-row 95% gain bound (5,000 draws), and city/stratified deterioration <=1%. These are historically examined labels and conditional robustness checks.

## Reproduction and compute

Elapsed charge at packaging: 1.3573 hours under the eight-hour cap, including setup and interruptions. Free T4 only; no portal upload.
V14 control reproduction must remain below INR 0.01. New candidate fresh/reload/query-layout/CSV agreement uses atol=10 INR and rtol=1e-6, plus at most INR 10 labeled-pilot RMSE drift. No output rounding is used.
The V7 portion retains the verified original-Mac fitted-state boundary. Qualified notebooks freshly fit the required TabPFN controls and new component. Expected predictions are verification witnesses after inference.
At most one manual portal trial is recommended after every gate passes. Keep the lowest observed submission and leave the one-time coding submission pending final model choice.

Sources: official Nums AI Causilo, NVIDIA structured-data-models, Layer6 TabDPT and Prior Labs TabPFN repositories. Exact sources, versions, hashes, failures and controls accompany the delivery.
