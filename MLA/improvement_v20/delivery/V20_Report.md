# V20 measured report

**Retain V14: no new V20 submission is recommended.**

Best observed portal RMSE: V14, 11,736,110.7877709. V20 has no portal score. Local validation is not a portal result.

## Fixed experiment

I8/I16: eight/sixteen-member TabICLv2. KS8-CLEAN: eight-member Kumo Small in an isolated environment without cuDF/Numba, with a fixed quantile-grid mean. The equal mean uses I8 and KS8-CLEAN when both pass engineering. Engineering admission is frozen before development scores.
Formula: (1-w)*V14+w*M, w in [0.10,0.25,0.50,0.75,1.00]. All 3,493 original labels remain in evaluation.

## Engineering

```json
{
  "status": "PASS",
  "settings": {
    "I8": {
      "status": "PASS",
      "fold_seconds": 15.419015972999887,
      "full_fit_predict_seconds": 10.249357077000013,
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
        "other_query_rows_changed_difference_INR": 0.0,
        "rows_checked": 699,
        "individual_probes": 12,
        "missing_unseen_inputs": true,
        "behavior_tolerance_INR": 10.0,
        "prediction_rtol": 1e-06,
        "rmse_tolerance_INR": 10.0
      },
      "fresh_repeat_difference_INR": 7.62939453125,
      "save_reload_difference_INR": 0.0,
      "target_metadata": {
        "name": "I8",
        "training_rows": 2794,
        "units": "raw INR millions",
        "model_class": "tabicl._sklearn.regressor.TabICLRegressor",
        "checkpoint_sha256": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
        "checkpoint_revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
        "actual_source_hashes": {
          "tabicl._sklearn.regressor": "9bfc15027a63211e7ab15a2924c13eb7537937dbd3d53e2799b18cdbc8d09731",
          "tabicl": "f40b23bee37298b180b86d3aed5444a0a9be154fd1e4b1cd02ee01b4bc112516"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.11.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "catboost": "1.2.10",
            "tabicl": "2.2.0",
            "structured-data-models": null,
            "numba": null,
            "cudf-cu13": null,
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1",
            "joblib": "1.6.0"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208,
          "cudf_discoverable": false,
          "numba_discoverable": false
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
        "peak_gpu_allocated_bytes": 2029224448,
        "peak_gpu_reserved_bytes": 2141192192,
        "host_rss_bytes": 1377071104,
        "query_layout": "Original V11 fixed 500 rows, padded with the training feature row having the most observed fields; no labels or IDs."
      },
      "full_target_metadata": {
        "name": "I8",
        "training_rows": 3493,
        "units": "raw INR millions",
        "model_class": "tabicl._sklearn.regressor.TabICLRegressor",
        "checkpoint_sha256": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
        "checkpoint_revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
        "actual_source_hashes": {
          "tabicl._sklearn.regressor": "9bfc15027a63211e7ab15a2924c13eb7537937dbd3d53e2799b18cdbc8d09731",
          "tabicl": "f40b23bee37298b180b86d3aed5444a0a9be154fd1e4b1cd02ee01b4bc112516"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.11.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "catboost": "1.2.10",
            "tabicl": "2.2.0",
            "structured-data-models": null,
            "numba": null,
            "cudf-cu13": null,
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1",
            "joblib": "1.6.0"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208,
          "cudf_discoverable": false,
          "numba_discoverable": false
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
        "peak_gpu_allocated_bytes": 2477515264,
        "peak_gpu_reserved_bytes": 2535456768,
        "host_rss_bytes": 1336725504,
        "query_layout": "Original V11 fixed 500 rows, padded with the training feature row having the most observed fields; no labels or IDs."
      },
      "peak_gpu_allocated_bytes": 2029224448,
      "peak_gpu_reserved_bytes": 2141192192,
      "training_rows": 2794,
      "validation_rows": 699,
      "full_training_rows": 3493,
      "selection_uses_validation_rmse": false,
      "engineering_seconds": 72.96325062300002
    },
    "KS8-CLEAN": {
      "status": "PASS",
      "fold_seconds": 19.919383167999968,
      "full_fit_predict_seconds": 11.172121404000109,
      "behavior": {
        "permutation_difference_INR": 0.0,
        "label_id_difference_INR": 0.0,
        "batch_difference_INR": 11.757448747754097,
        "individual_difference_INR": 13.317551098763943,
        "rmse_difference_INR": {
          "permutation": 0.0,
          "batch": 0.24964402616024017,
          "individual": 0.7099003307521343
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
        "name": "KS8-CLEAN",
        "training_rows": 2794,
        "units": "raw INR millions",
        "model_class": "sdm.models.kumo.tabular.model.KumoTabular",
        "checkpoint_sha256": "fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423",
        "checkpoint_revision": "3c3e10bbdb590ace29e7026847f92db3c603096d",
        "actual_source_hashes": {
          "sdm.models.kumo.tabular.model": "fcd1ace9e011dadcf152abae2608196ca3d1ca5dab470a1384931f794a97d882",
          "sdm.models.kumo.tabular.recipe": "a5af156ffcb82cadb8e67070bbf1230c29b3fe245bb8ed89cebcbeaf738f77a0",
          "sdm.models.kumo.tabular.icl": "8004617d373d8f350300d1f85c3883936a5c8da0871781e29707754370231c7d",
          "sdm.models.kumo.tabular.batching": "4b9eded8f1b3fbc175a2d1e3ccf964c2a368378b15abd3e17ca48b51a19c31af",
          "sdm.models.base.icl": "7a88f8c260061284424982c4484889e03dd84e2e0efaf7f90596aea8f405ef64"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.11.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "catboost": "1.2.10",
            "tabicl": null,
            "structured-data-models": "0.1.0.dev0",
            "numba": null,
            "cudf-cu13": null,
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1",
            "joblib": "1.6.0"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208,
          "cudf_discoverable": false,
          "numba_discoverable": false
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
        "peak_gpu_allocated_bytes": 950044672,
        "peak_gpu_reserved_bytes": 2178940928,
        "host_rss_bytes": 3642195968
      },
      "full_target_metadata": {
        "name": "KS8-CLEAN",
        "training_rows": 3493,
        "units": "raw INR millions",
        "model_class": "sdm.models.kumo.tabular.model.KumoTabular",
        "checkpoint_sha256": "fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423",
        "checkpoint_revision": "3c3e10bbdb590ace29e7026847f92db3c603096d",
        "actual_source_hashes": {
          "sdm.models.kumo.tabular.model": "fcd1ace9e011dadcf152abae2608196ca3d1ca5dab470a1384931f794a97d882",
          "sdm.models.kumo.tabular.recipe": "a5af156ffcb82cadb8e67070bbf1230c29b3fe245bb8ed89cebcbeaf738f77a0",
          "sdm.models.kumo.tabular.icl": "8004617d373d8f350300d1f85c3883936a5c8da0871781e29707754370231c7d",
          "sdm.models.kumo.tabular.batching": "4b9eded8f1b3fbc175a2d1e3ccf964c2a368378b15abd3e17ca48b51a19c31af",
          "sdm.models.base.icl": "7a88f8c260061284424982c4484889e03dd84e2e0efaf7f90596aea8f405ef64"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.11.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "catboost": "1.2.10",
            "tabicl": null,
            "structured-data-models": "0.1.0.dev0",
            "numba": null,
            "cudf-cu13": null,
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1",
            "joblib": "1.6.0"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208,
          "cudf_discoverable": false,
          "numba_discoverable": false
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
        "peak_gpu_allocated_bytes": 1145169920,
        "peak_gpu_reserved_bytes": 2818572288,
        "host_rss_bytes": 3554852864
      },
      "peak_gpu_allocated_bytes": 950044672,
      "peak_gpu_reserved_bytes": 2178940928,
      "training_rows": 2794,
      "validation_rows": 699,
      "full_training_rows": 3493,
      "selection_uses_validation_rmse": false,
      "engineering_seconds": 85.09807243
    },
    "I16": {
      "status": "PASS",
      "fold_seconds": 22.22361828199996,
      "full_fit_predict_seconds": 14.571914581000101,
      "behavior": {
        "permutation_difference_INR": 7.62939453125,
        "label_id_difference_INR": 0.0,
        "batch_difference_INR": 7.62939453125,
        "individual_difference_INR": 0.0,
        "rmse_difference_INR": {
          "permutation": 0.008478622883558273,
          "batch": 0.02872493863105774,
          "individual": 0.0
        },
        "other_query_rows_changed_difference_INR": 0.0,
        "rows_checked": 699,
        "individual_probes": 12,
        "missing_unseen_inputs": true,
        "behavior_tolerance_INR": 10.0,
        "prediction_rtol": 1e-06,
        "rmse_tolerance_INR": 10.0
      },
      "fresh_repeat_difference_INR": 15.2587890625,
      "save_reload_difference_INR": 0.0,
      "target_metadata": {
        "name": "I16",
        "training_rows": 2794,
        "units": "raw INR millions",
        "model_class": "tabicl._sklearn.regressor.TabICLRegressor",
        "checkpoint_sha256": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
        "checkpoint_revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
        "actual_source_hashes": {
          "tabicl._sklearn.regressor": "9bfc15027a63211e7ab15a2924c13eb7537937dbd3d53e2799b18cdbc8d09731",
          "tabicl": "f40b23bee37298b180b86d3aed5444a0a9be154fd1e4b1cd02ee01b4bc112516"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.11.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "catboost": "1.2.10",
            "tabicl": "2.2.0",
            "structured-data-models": null,
            "numba": null,
            "cudf-cu13": null,
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1",
            "joblib": "1.6.0"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208,
          "cudf_discoverable": false,
          "numba_discoverable": false
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
        "peak_gpu_allocated_bytes": 3885072896,
        "peak_gpu_reserved_bytes": 4099932160,
        "host_rss_bytes": 1340784640,
        "query_layout": "Original V11 fixed 500 rows, padded with the training feature row having the most observed fields; no labels or IDs."
      },
      "full_target_metadata": {
        "name": "I16",
        "training_rows": 3493,
        "units": "raw INR millions",
        "model_class": "tabicl._sklearn.regressor.TabICLRegressor",
        "checkpoint_sha256": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
        "checkpoint_revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
        "actual_source_hashes": {
          "tabicl._sklearn.regressor": "9bfc15027a63211e7ab15a2924c13eb7537937dbd3d53e2799b18cdbc8d09731",
          "tabicl": "f40b23bee37298b180b86d3aed5444a0a9be154fd1e4b1cd02ee01b4bc112516"
        },
        "runtime": {
          "dependencies": {
            "torch": "2.11.0",
            "numpy": "2.5.4",
            "pandas": "3.0.6",
            "scikit-learn": "1.9.1",
            "scipy": "1.18.1",
            "catboost": "1.2.10",
            "tabicl": "2.2.0",
            "structured-data-models": null,
            "numba": null,
            "cudf-cu13": null,
            "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0",
            "pyarrow": "23.0.1",
            "joblib": "1.6.0"
          },
          "gpu": "Tesla T4",
          "cuda": "13.0",
          "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
          "total_gpu_bytes": 15637086208,
          "cudf_discoverable": false,
          "numba_discoverable": false
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
        "peak_gpu_allocated_bytes": 4728883712,
        "peak_gpu_reserved_bytes": 4932501504,
        "host_rss_bytes": 1332604928,
        "query_layout": "Original V11 fixed 500 rows, padded with the training feature row having the most observed fields; no labels or IDs."
      },
      "peak_gpu_allocated_bytes": 3885072896,
      "peak_gpu_reserved_bytes": 4099932160,
      "training_rows": 2794,
      "validation_rows": 699,
      "full_training_rows": 3493,
      "selection_uses_validation_rmse": false,
      "engineering_seconds": 118.549284682
    }
  },
  "screen_settings": [
    "I8",
    "KS8-CLEAN",
    "I16"
  ],
  "routes": {
    "I8": {
      "I8": 1.0
    },
    "KS8-CLEAN": {
      "KS8-CLEAN": 1.0
    },
    "I16": {
      "I16": 1.0
    },
    "CORE_EQUAL": {
      "I8": 0.5,
      "KS8-CLEAN": 0.5
    }
  },
  "engineering_feasible_settings": [
    "I8",
    "KS8-CLEAN",
    "I16"
  ],
  "budget_exclusions": {},
  "screen_forecast_seconds": 748.3062264989976,
  "confirmation_forecast_seconds": 5996.89714707854,
  "full_data_and_fresh_notebook_forecast_seconds": 2395.6958440506005,
  "forecast_fits_budget": true,
  "selection_uses_validation_rmse": false,
  "sources": {
    "core.py": "4dde23640f57f113cf36bb02a5d988ba66c4551c5ee20b656677078b5e289b81",
    "features.py": "a04a984725d4e6c9a8a2a9b21f5a154970ffb399cda5fbd578562a4e031204ac",
    "modeling.py": "bb17ffb4d8fe38485dd2f405a411c10f53ff96a4c21ee6810767d154e4710aa5",
    "workflow.py": "b256b89bd8aa0e17c59b7fed5e2edeff1bde796dcc32ebde47e5238967038983",
    "verify.py": "e5e73c2beea5a4d217f56b21c6a026967d01bcc6b609db251cee222cee04e53d",
    "delivery.py": "0591f783f1223c09be5988dafbce7d541939b040687b580666478cb5a0465427",
    "fresh_worker.py": "3b7e3529fc117f18fb8c85af9d434931d673fd5873015355be89e9190c905f73",
    "baseline.py": "0513655049b5d7c0e4e57b3339596e718779cd7bcbcc49bafd76e14e5bd9dad1",
    "legacy_models.py": "031c25ba5a60cf3e6e4c8247553ee951ff71742c846eb103246140929f6cdfba",
    "legacy_reference.py": "1a24f9f116170ea54bd3242264d5cd2a150c3429e039a50cb11c3112e52247c1",
    "frozen_v9_modeling.py": "c4d948bcf9e1aab5e667283eb82889af44bc57147a0c2a234af6322044b55487",
    "v5_contract.json": "3446412f01d536a31123142b984392216efd29094965500251386eac1814ab69",
    "city_map.json": "539387a0a9f5e143228714dde7679c2848a019a1cbbaa87129b0d4e4c092f7f9",
    "reference_manifest.json": "82417467e331d6e470fd1058c4ea4edb5a47f98021efc822176bd9a04b99bbd5",
    "baseline_registry.json": "53e7ddb7fbfbe08162aa97fac8d5ad9dea4981aab3aa942b3e98ebdb2a3060b7",
    "frozen_pfn.py": "ff4d383d269ffdc1b0021f7d0cac4e19240be130f2544bcc80dc41916a5262b5",
    "new_models.py": "d503b570ae3b1077555f43ccb41b3a43d015db7fd2a0e141a623a8264625d3ca",
    "candidate_models.py": "3cfdf09afd25eb4b19bde4632433c67b09a759117f8f1db1c86227cb610c6b32",
    "candidate_worker.py": "a70cb7f5406a194eacbd7e653a84c0c5f60480fa602799be31993ff8049c9e0d",
    "pilot_worker.py": "b41a2f96fdbaaa8ce17426bad73be48c0f66151f4a110549bf2f680b172fbd28",
    "checkpoint_registry.json": "c2285e8bb17a1bb162aaef60e11992af7c140b60528f8df0c780a5fa7ae12d78",
    "source_registry.json": "24436f72ac9c9eb3c6706ad1dfecf414beff26480be9682ed170b1397a54c740",
    "protocol_proposed.json": "81a0867385dfa14e343b2a6ff88ca807525c924935cc93db7ce595ceeacc625f",
    "mac_controls.py": "c6c01ea36a0a9f3a4c5a6e4c8110c4e07b83e5373c110e075f4a7844515208c3",
    "control_worker.py": "79c40d599b5aff4a3a120464723d9fd1075200195787d7e4e157827e10945fca",
    "setup_environment.py": "09c98fd69c8913bc10663df71633e9bbb7adf53b95f391fabb65c2f5ae30801c",
    "runtime_lock.json": "b3a4f31c5a6cd6bb08a8af3850b9c32ceed57c7b4c2d34ca8f4416dbc72466cc",
    "candidate_dependency_locks.json": "513662be85660d02739003c79ebd4a612ff4bc1f846cbf4ec694aa08aa60395a"
  },
  "inputs": {
    "Train_dataset.csv": "ec40ebda74ba0bfbfe1af4417a4266a70eadf6e595500e99e5435e7bdc7ee51f",
    "Test_dataset.csv": "ee4137566adb1c0ce356a8c370e7b8a22f681bde492068efe11346d8d44b4f4d",
    "Data_Dictionary.csv": "2d86c00e27448593564a161197b6639ee1eb90c01fa523b67e7a09d2ef405c5e",
    "Sample_Solution.csv": "9519ba4d51eb2d59b8a71b054a62e1f4d48b72cf92930c31f5e5393a2cd135fb"
  },
  "model_contract_sha256": "56d9e243f598a89a1ed97408f5c8b62a8ed05e27ff17f65e2370736dae6d79b8",
  "control_timings": {
    "BASE": {
      "status": "PASS",
      "seconds": 132.27559460400005,
      "maximum_existing_fold_difference_INR": 0.0
    },
    "P35_8": {
      "status": "PASS",
      "seconds": 35.15834585799996,
      "maximum_existing_fold_difference_INR": 0.0
    }
  },
  "original_mac_control_timing": {
    "status": "PASS",
    "seconds": 81.20191466600227,
    "maximum_difference_INR": 1.4901161193847656e-08
  }
}
```

## Development

| Setting | Weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|---:|---:|
| I8 | 10% | 19,647,141.66 | 19,652,156.43 | 1,891.90 | -2,250.98 | -0.00091% |
| I8 | 25% | 19,652,239.86 | 19,664,684.57 | -3,206.31 | -14,779.12 | -0.04577% |
| I8 | 50% | 19,681,872.84 | 19,709,906.21 | -32,839.28 | -60,000.76 | -0.23624% |
| I8 | 75% | 19,737,813.47 | 19,785,361.64 | -88,779.91 | -135,456.18 | -0.57059% |
| I8 | 100% | 19,819,838.99 | 19,890,706.78 | -170,805.43 | -240,801.33 | -1.04737% |
| KS8-CLEAN | 10% | 19,649,585.29 | 19,650,139.06 | -551.74 | -233.61 | -0.00200% |
| KS8-CLEAN | 25% | 19,652,344.99 | 19,652,514.57 | -3,311.44 | -2,609.11 | -0.01507% |
| KS8-CLEAN | 50% | 19,662,093.83 | 19,661,871.23 | -13,060.28 | -11,965.78 | -0.06368% |
| KS8-CLEAN | 75% | 19,678,270.50 | 19,677,965.83 | -29,236.95 | -28,060.37 | -0.14580% |
| KS8-CLEAN | 100% | 19,700,859.17 | 19,700,781.84 | -51,825.62 | -50,876.38 | -0.26134% |
| I16 | 10% | 19,646,951.35 | 19,650,950.64 | 2,082.20 | -1,045.19 | 0.00264% |
| I16 | 25% | 19,651,561.85 | 19,661,289.37 | -2,528.29 | -11,383.92 | -0.03540% |
| I16 | 50% | 19,679,845.25 | 19,701,858.46 | -30,811.69 | -51,953.00 | -0.21060% |
| I16 | 75% | 19,733,773.02 | 19,771,433.06 | -84,739.46 | -121,527.61 | -0.52486% |
| I16 | 100% | 19,813,135.76 | 19,869,708.49 | -164,102.21 | -219,803.04 | -0.97688% |
| CORE_EQUAL | 10% | 19,647,855.59 | 19,650,576.97 | 1,177.96 | -671.51 | 0.00129% |
| CORE_EQUAL | 25% | 19,649,118.33 | 19,655,034.04 | -84.77 | -5,128.59 | -0.01327% |
| CORE_EQUAL | 50% | 19,659,299.06 | 19,671,653.43 | -10,265.50 | -21,747.98 | -0.08146% |
| CORE_EQUAL | 75% | 19,679,560.08 | 19,699,734.54 | -30,526.53 | -49,829.08 | -0.20447% |
| CORE_EQUAL | 100% | 19,709,870.32 | 19,739,228.44 | -60,836.76 | -89,322.99 | -0.38209% |

## Decision and validation

Retain V14: no development candidate qualified.
Confirmation, bootstrap qualification, a new CSV and a fresh V20 submission notebook were skipped because no development candidate qualified. Byte-identical scored V14 files are preserved under KEEP_V14 names. Engineering exclusions are not accuracy failures.

Require positive gains on both development assignments and mean gain >=0.20% versus V14. Freeze one finalist before confirmation. Require all three random confirmation gains positive, mean relative gain >=0.20%, positive lower paired-row 95% gain bound (5,000 draws), and city/stratified deterioration <=1%. These are historically examined labels and conditional robustness checks.

## Reproduction and compute

Elapsed charge at packaging: 0.9654 hours under the six-hour cap, including setup and interruptions. Free T4 only; no portal upload.
V14 control reproduction must remain below INR 0.01. New candidate fresh/reload/query-layout/CSV agreement uses atol=10 INR and rtol=1e-6, plus at most INR 10 labeled-pilot RMSE drift. No output rounding is used.
The V7 portion retains the verified original-Mac fitted-state boundary. Qualified notebooks freshly fit the required TabPFN controls and new component. Expected predictions are verification witnesses after inference.
At most one manual portal trial is recommended after every gate passes. Keep the lowest observed submission and leave the one-time coding submission pending final model choice.

Sources: official soda-inria TabICL, NVIDIA structured-data-models and Prior Labs TabPFN repositories. Exact sources, versions, hashes, failures and controls accompany the delivery.
