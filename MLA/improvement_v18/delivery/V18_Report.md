# V18 measured report

**Retain V14: no new V18 submission is recommended.**

Best observed portal RMSE: V14, 11,736,110.7877709. V18 has no portal score. Local validation is not a portal result.

## Fixed experiment

FT512 and FT1024: constant-rate 3e-6 TabPFN-3.5 adaptation at two fixed context sizes. EXA8: eight-member float32 EXAONE with mean readout and uniform weighting. Two equal FT/EXA mixtures are predeclared. Engineering admission is frozen before development scores.
Formula: (1-w)*V14+w*M, w in [0.10,0.25,0.50,0.75,1.00]. All 3,493 original labels remain in evaluation.

## Engineering

```json
{
  "status": "PASS",
  "settings": {
    "FT512": {
      "status": "PASS",
      "fold_seconds": 316.4080967296359,
      "full_fit_predict_seconds": 390.5628783736703,
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
        "name": "FT512",
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
            "mse_millions_squared": 257.73304344177683
          },
          {
            "epoch": 2,
            "mse_millions_squared": 257.7352424653391
          },
          {
            "epoch": 3,
            "mse_millions_squared": 257.2273356884834
          },
          {
            "epoch": 4,
            "mse_millions_squared": 257.77608703972226
          },
          {
            "epoch": 5,
            "mse_millions_squared": 258.79696040973283
          },
          {
            "epoch": 6,
            "mse_millions_squared": 259.8178175858402
          },
          {
            "epoch": 7,
            "mse_millions_squared": 260.66654442850415
          },
          {
            "epoch": 8,
            "mse_millions_squared": 261.5643458483341
          },
          {
            "epoch": 9,
            "mse_millions_squared": 262.24794906467395
          },
          {
            "epoch": 10,
            "mse_millions_squared": 262.67191368577375
          },
          {
            "epoch": 11,
            "mse_millions_squared": 262.986724311396
          }
        ],
        "outer_validation_used_for_epoch_selection": false,
        "precision": "CUDA-AMP-training/float32-inference",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "matched_original_architectures": [
          "tabpfn.architectures.tabpfn_v3_5.TabPFNV3p5"
        ],
        "installed_finetune_source_sha256": "315413cd044720d1edc9962d6697ed55cea90f3e7181d1a2a1c41ac8ccde18fa",
        "learning_rate": 3e-06,
        "use_lr_scheduler": false,
        "setup_seconds": 30.939502929000014,
        "selection_seconds": 62.734027207000054,
        "refit_seconds": 11.32334549199993,
        "inner_optimizer_steps": 44,
        "full_subset_refitted": true,
        "actual_estimators": 1,
        "context_plus_query_rows": 512,
        "matched_unadapted_inner_mse_millions_squared": 257.5474534175994,
        "fit_seconds": 108.134125015,
        "last_query_seconds": 1.1423374070000136
      },
      "peak_gpu_allocated_bytes": 5208716288,
      "training_rows": 2794,
      "validation_rows": 699,
      "full_training_rows": 3493,
      "selection_uses_validation_rmse": false,
      "engineering_seconds": 334.084323576,
      "forecast_basis": "Measured fixed setup, maximum thirty selection epochs, thirty refit epochs, full validation and final queries; thirty-percent contingency applied by admission."
    },
    "EXA8": {
      "status": "engineering_excluded",
      "error_type": "RuntimeError",
      "reason": "No available kernel. Aborting execution.",
      "traceback": "Traceback (most recent call last):
  File \"/content/mla_v18/pilot_worker.py\", line 28, in main
    p=modeling.predict(fitted,e.train.iloc[b])
  File \"/content/mla_v18/modeling.py\", line 50, in predict
    return new_models.predict(fitted,raw)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File \"/content/mla_v18/new_models.py\", line 184, in predict
    if fitted['name']=='EXA8':p=fitted['model'].predict(block)
                                ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/regressor.py\", line 443, in predict
    member_predictions = self._pooled_member_points(
        support_x, support_y, query_x, state[\"passes\"],
        preprocessors=state[\"preprocessors\"],
    )
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/regressor.py\", line 268, in _pooled_member_points
    self._member_points(
    ~~~~~~~~~~~~~~~~~~~^
        support_x, support_y, query_x, n_svd=n_svd, seed=seed,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        preprocessor=None if preprocessors is None else preprocessors[seed],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        categorical_mask=categorical_mask,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/regressor.py\", line 234, in _member_points
    output = self._member_quantiles(
        support_x, support_y, query_x, n_svd=n_svd, seed=seed,
        preprocessor=preprocessor, categorical_mask=categorical_mask,
    )
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/regressor.py\", line 224, in _member_quantiles
    output = self._forward_chunked(batch_xs, batch_y, batch_xq)
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/regressor.py\", line 159, in _forward_chunked
    return self._executor().forward(
           ~~~~~~~~~~~~~~~~~~~~~~~~^
        self.model, support_batch, label_batch, query_batch
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/_execution.py\", line 837, in forward
    return self._run_execution_plan(
           ~~~~~~~~~~~~~~~~~~~~~~~~^
        model, support_batch, label_batch, query_batch, execution_plan
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/_execution.py\", line 866, in _run_execution_plan
    model(
    ~~~~~^
        member_support,
        ^^^^^^^^^^^^^^^
    ...<10 lines>...
        trusted_internal_inputs=True,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File \"/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py\", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File \"/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py\", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/model/heads.py\", line 135, in forward
    return self.transformer(
           ~~~~~~~~~~~~~~~~^
        state,
        ^^^^^^
    ...<6 lines>...
        query_chunk_size=query_chunk_size,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File \"/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py\", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File \"/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py\", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/model/transformer.py\", line 307, in forward
    state, feature_summaries, item_summaries = layer(
                                               ~~~~~^
        state,
        ^^^^^^
    ...<9 lines>...
        output_support=output == \"all\" or not layer.final_layer,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File \"/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py\", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File \"/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py\", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/model/layer.py\", line 489, in forward
    combined = block(state, item_summary)
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/model/layer.py\", line 393, in _feature_attention_block
    mixed = self._attention(
            ~~~~~~~~~~~~~~~^
        self.feature_context_attentions[repeat], query, context
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ).transpose(1, 2)
    ^
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/model/layer.py\", line 263, in _attention
    return module(
        query,
    ...<3 lines>...
        reuse_first_context_head=shared_context_head,
    )
  File \"/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py\", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File \"/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py\", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File \"/content/mla_v18/engineering_sources/exaone/src/exaonetabular/model/attention.py\", line 402, in forward
    attended = F.scaled_dot_product_attention(
        flat_query,
    ...<2 lines>...
        **attention_options,
    )
RuntimeError: No available kernel. Aborting execution.
",
      "engineering_seconds": 11.003430449000007,
      "accuracy_tested": false
    },
    "FT1024": {
      "status": "PASS",
      "fold_seconds": 403.9734352782851,
      "full_fit_predict_seconds": 286.7583015084994,
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
        "name": "FT1024",
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
        "inner_training_rows": 2374,
        "inner_validation_rows": 420,
        "inner_seed": 9091,
        "inner_train_sha256": "7f75c6d798f5f3e1aec1348dcd7f28ed151479de37588d9ef49616d76d44b982",
        "inner_validation_sha256": "78b66a263f116bce32eb4da28de7e21795134e6708d165edc8847f8b66befade",
        "inner_policy": "random-15-percent",
        "selected_epochs": 7,
        "inner_epochs_run": 15,
        "inner_history": [
          {
            "epoch": 0,
            "mse_millions_squared": 257.5474534175994
          },
          {
            "epoch": 1,
            "mse_millions_squared": 257.4536369700709
          },
          {
            "epoch": 2,
            "mse_millions_squared": 257.4536369700709
          },
          {
            "epoch": 3,
            "mse_millions_squared": 257.5976362853548
          },
          {
            "epoch": 4,
            "mse_millions_squared": 257.2842649358762
          },
          {
            "epoch": 5,
            "mse_millions_squared": 256.88723366269085
          },
          {
            "epoch": 6,
            "mse_millions_squared": 256.47544345805835
          },
          {
            "epoch": 7,
            "mse_millions_squared": 256.309166054754
          },
          {
            "epoch": 8,
            "mse_millions_squared": 256.35503008854766
          },
          {
            "epoch": 9,
            "mse_millions_squared": 256.55377280179636
          },
          {
            "epoch": 10,
            "mse_millions_squared": 257.11249256672414
          },
          {
            "epoch": 11,
            "mse_millions_squared": 257.88836490112124
          },
          {
            "epoch": 12,
            "mse_millions_squared": 258.68362151878233
          },
          {
            "epoch": 13,
            "mse_millions_squared": 259.2854202803532
          },
          {
            "epoch": 14,
            "mse_millions_squared": 259.58502839025186
          },
          {
            "epoch": 15,
            "mse_millions_squared": 259.932731278755
          }
        ],
        "outer_validation_used_for_epoch_selection": false,
        "precision": "CUDA-AMP-training/float32-inference",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "matched_original_architectures": [
          "tabpfn.architectures.tabpfn_v3_5.TabPFNV3p5"
        ],
        "installed_finetune_source_sha256": "315413cd044720d1edc9962d6697ed55cea90f3e7181d1a2a1c41ac8ccde18fa",
        "learning_rate": 3e-06,
        "use_lr_scheduler": false,
        "setup_seconds": 7.429217434000066,
        "selection_seconds": 154.44424377699988,
        "refit_seconds": 20.166661116999876,
        "inner_optimizer_steps": 30,
        "full_subset_refitted": true,
        "actual_estimators": 1,
        "context_plus_query_rows": 1024,
        "matched_unadapted_inner_mse_millions_squared": 257.5474534175994,
        "fit_seconds": 185.53170100300008,
        "last_query_seconds": 1.2271826460000739
      },
      "peak_gpu_allocated_bytes": 5210158080,
      "training_rows": 2794,
      "validation_rows": 699,
      "full_training_rows": 3493,
      "selection_uses_validation_rmse": false,
      "engineering_seconds": 510.20331741500013,
      "forecast_basis": "Measured fixed setup, maximum thirty selection epochs, thirty refit epochs, full validation and final queries; thirty-percent contingency applied by admission."
    }
  },
  "screen_settings": [
    "FT512",
    "FT1024"
  ],
  "engineering_feasible_settings": [
    "FT512",
    "FT1024"
  ],
  "budget_exclusions": {},
  "screen_forecast_seconds": 9487.163630982974,
  "confirmation_forecast_seconds": 18425.55260552128,
  "full_data_and_fresh_notebook_forecast_seconds": 3355.4634837715425,
  "forecast_fits_budget": true,
  "selection_uses_validation_rmse": false,
  "sources": {
    "core.py": "466945c93654077fc6599700a73a7fdb23bdd415893e66635770dd7b3a1b086f",
    "features.py": "86939252439cba79a803e1cc3563ecc83e5ff37dc4ab52b79edb71e3fd22fe1c",
    "modeling.py": "6380180020158744654788e41d44af0ef1518c6f1fa130b1c3ffcca642f1ff46",
    "workflow.py": "d0910de2bef1aa3397007921d67580d1060f413719189d09cccf44c194a1985a",
    "verify.py": "8032e62e6eb8a0770ccf11eb324dd9640ef87d6ffb983c3e7d094858e3e7172e",
    "delivery.py": "89459ed273dc457d6d517f4fcc13525e064fccd7c2b01aafdf8ccc2e7bfbd4c6",
    "fresh_worker.py": "2bf9f9fadc22bf0b4db26a39fbb597da68ec5ded129efa9a8d26f525da619cd0",
    "baseline.py": "0513655049b5d7c0e4e57b3339596e718779cd7bcbcc49bafd76e14e5bd9dad1",
    "legacy_models.py": "031c25ba5a60cf3e6e4c8247553ee951ff71742c846eb103246140929f6cdfba",
    "legacy_reference.py": "1a24f9f116170ea54bd3242264d5cd2a150c3429e039a50cb11c3112e52247c1",
    "frozen_v9_modeling.py": "c4d948bcf9e1aab5e667283eb82889af44bc57147a0c2a234af6322044b55487",
    "v5_contract.json": "3446412f01d536a31123142b984392216efd29094965500251386eac1814ab69",
    "city_map.json": "539387a0a9f5e143228714dde7679c2848a019a1cbbaa87129b0d4e4c092f7f9",
    "reference_manifest.json": "82417467e331d6e470fd1058c4ea4edb5a47f98021efc822176bd9a04b99bbd5",
    "baseline_registry.json": "53e7ddb7fbfbe08162aa97fac8d5ad9dea4981aab3aa942b3e98ebdb2a3060b7",
    "frozen_pfn.py": "ff4d383d269ffdc1b0021f7d0cac4e19240be130f2544bcc80dc41916a5262b5",
    "new_models.py": "2cc4fef5221d8cb4a41421760687679b5a83c66248eb0d36a60253b44c68c48e",
    "pilot_worker.py": "f9721329d6fcc6f8d395f175d6fd24ed954dcd9cf5533db655e1e66191a2bc6e",
    "checkpoint_registry.json": "4dd0b5e632d00b688e6dca18892c46a80512eafbd2f5b783850d66a0f76a233c",
    "source_registry.json": "4ddfe7ed70327ced487ee6b6ada889e37dbfc81c4785e4cbe7c1fd5fa7faf81a",
    "protocol_proposed.json": "9753ec10e1b15210b12240b89c1609eaf65a6afd527fc657bcd684f31e2445f5",
    "release_review_registry.json": "ba98e8ecee9cb9188b8ce87a8e607181c249c89727bfab2a63985b0267a8a470",
    "mac_controls.py": "c6c01ea36a0a9f3a4c5a6e4c8110c4e07b83e5373c110e075f4a7844515208c3",
    "control_worker.py": "4b5abe58acd90cedb8c2557166fda48a0daff2e8b75da67a87aa975678d61ee3"
  },
  "inputs": {
    "Train_dataset.csv": "ec40ebda74ba0bfbfe1af4417a4266a70eadf6e595500e99e5435e7bdc7ee51f",
    "Test_dataset.csv": "ee4137566adb1c0ce356a8c370e7b8a22f681bde492068efe11346d8d44b4f4d",
    "Data_Dictionary.csv": "2d86c00e27448593564a161197b6639ee1eb90c01fa523b67e7a09d2ef405c5e",
    "Sample_Solution.csv": "9519ba4d51eb2d59b8a71b054a62e1f4d48b72cf92930c31f5e5393a2cd135fb"
  },
  "control_timings": {
    "BASE": {
      "status": "PASS",
      "seconds": 122.99813944400012,
      "maximum_existing_fold_difference_INR": 0.0
    },
    "P35_8": {
      "status": "PASS",
      "seconds": 44.987363232999996,
      "maximum_existing_fold_difference_INR": 0.0
    },
    "UNADAPTED": {
      "status": "PASS",
      "seconds": 9.400285760000088,
      "maximum_existing_fold_difference_INR": null
    }
  },
  "mac_control_pilot": {
    "status": "PASS",
    "seconds": 87.95842920900031,
    "maximum_difference_INR": 1.4901161193847656e-08
  },
  "unadapted_seconds": 9.400285760000088,
  "new_runtime": {
    "safetensors": "0.8.0",
    "huggingface-hub": "1.33.0",
    "nbformat": "5.11.1",
    "nbclient": "0.10.4"
  },
  "model_contract_sha256": "d9b2459075f3e8b4488dceba3960cbc6fc4da67577c93a7cc7912bb5767462f3"
}
```

## Development

| Setting | Weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|---:|---:|
| FT512 | 10% | 19,651,703.12 | 19,641,259.74 | -2,669.57 | 8,645.71 | 0.01521% |
| FT512 | 25% | 19,656,913.06 | 19,629,745.65 | -7,879.50 | 20,159.81 | 0.03125% |
| FT512 | 50% | 19,668,808.12 | 19,614,441.41 | -19,774.57 | 35,464.04 | 0.03992% |
| FT512 | 75% | 19,684,711.48 | 19,604,004.11 | -35,677.92 | 45,901.34 | 0.02601% |
| FT512 | 100% | 19,704,613.41 | 19,598,441.53 | -55,579.85 | 51,463.93 | -0.01048% |
| FT1024 | 10% | 19,651,677.66 | 19,643,803.53 | -2,644.10 | 6,101.93 | 0.00880% |
| FT1024 | 25% | 19,656,827.42 | 19,635,927.12 | -7,793.87 | 13,978.33 | 0.01574% |
| FT1024 | 50% | 19,668,563.73 | 19,626,208.07 | -19,530.18 | 23,697.38 | 0.01060% |
| FT1024 | 75% | 19,684,235.44 | 19,620,754.62 | -35,201.88 | 29,150.83 | -0.01540% |
| FT1024 | 100% | 19,703,833.15 | 19,619,570.34 | -54,799.59 | 30,335.12 | -0.06226% |

## Decision and validation

Retain V14: no development candidate qualified.
Confirmation, bootstrap qualification, a new CSV and a fresh V18 submission notebook were skipped because no development candidate qualified. Byte-identical scored V14 files are preserved under KEEP_V14 names. Engineering exclusions are not accuracy failures.

Require positive gains on both development assignments and mean gain >=0.20% versus V14. Freeze one finalist before confirmation. Require all three random confirmation gains positive, mean relative gain >=0.20%, positive lower paired-row 95% gain bound (5,000 draws), and city/stratified deterioration <=1%. These are historically examined labels and conditional robustness checks.

## Reproduction and compute

Elapsed charge at packaging: 2.1414 hours under the eighteen-hour cap, including setup and interruptions. Free T4 only; no portal upload.
Same-layout fresh reproduction and CSV agreement must be below INR 0.01. Permutation/batch/individual probes permit INR 10 prediction difference and INR 1 pilot RMSE change. No output rounding is used.
The V7 portion retains the verified original-Mac fitted-state boundary. Qualified notebooks freshly fit the required TabPFN controls and new component. Expected predictions are verification witnesses after inference.
At most one manual portal trial is recommended after every gate passes. Keep the lowest observed submission and leave the one-time coding submission pending final model choice.

Sources: official LG AI Research EXAONE and Prior Labs TabPFN repositories. Exact sources, versions, hashes, failures and controls accompany the delivery.
