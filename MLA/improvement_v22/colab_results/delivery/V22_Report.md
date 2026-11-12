# V22 measured report

**Retain V14: no new V22 submission is recommended.**

Best observed portal RMSE: V14, 11,736,110.7877709. V22 has not been uploaded. Local validation is not a portal score.

This is one authorized fixed survey-pair ablation. The four-hour operational ledger includes implementation, setup, failures and interruptions; reserve 1.5 hours for delivery. Earlier version ledgers remain unchanged.

## Fixed experiment

One setting: P35-SC38 (38 features), the exact original V14 forty-column view with Service and Comfortablility removed. Use the exact V14 TabPFN-3.5 checkpoint, eight estimators, seed 10242 and all original labels in INR millions. Categorical vocabularies fit on training rows only. Checkpoint temperature/calibration defaults are unchanged. Pair removal existed for older tree recipes; this is its controlled transfer to the P35 backbone.

Formula: .25*V9 + .75*((1-w)*original_P35 + w*new_component), w in [.25,.5,1]. Three comparisons maximum. Baseline is scored V14.

## Engineering pilots

```json
{
  "status": "PASS",
  "settings": {
    "P35-SC38": {
      "status": "PASS",
      "fold_seconds": 22.459282578,
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
        "rmse_tolerance_INR": 1.0,
        "removed_feature_mutation_difference_INR": 0.0
      },
      "fresh_repeat_difference_INR": 0.0,
      "target_metadata": {
        "name": "P35-SC38",
        "training_rows": 2794,
        "units": "raw INR millions",
        "checkpoint_sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
        "model_class": "tabpfn.regressor.TabPFNRegressor",
        "view": "SC38",
        "input_features": 38,
        "temperature": "checkpoint default",
        "feature_state_sha256": "c4290f5e843eaf83d34f68b332db7f3f309848e17b5c0684c7f25d520a5300c6",
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
      "full_data_seconds": 15.331928289000075,
      "full_data_training_rows": 3493,
      "full_data_query_rows": 500,
      "reload_difference_INR": 0.0,
      "peak_gpu_allocated_bytes": 8052313600,
      "training_rows": 2794,
      "validation_rows": 699,
      "engineering_seconds": 287.1774888670001
    }
  },
  "screen_settings": [
    "P35-SC38"
  ],
  "engineering_feasible_settings": [
    "P35-SC38"
  ],
  "budget_exclusions": {},
  "screen_forecast_seconds": 291.970673514,
  "confirmation_forecast_seconds": 729.9266837850001,
  "delivery_forecast_seconds": 509.58904065420063,
  "forecast_fits_budget": true,
  "selection_uses_validation_rmse": false,
  "sources": {
    "core.py": "778305bc851f53733452339e1e448af2ed6fb124cc47040bbdd6e0b1ce4ba1a5",
    "features.py": "4294846f13ace69e64914955117ff0f629ce64927771db479e30ecd4f03d2fa1",
    "modeling.py": "bfb1a0f610f3dc3df416121a8a384d96044cefc3b5cca99cce447761a6e21149",
    "workflow.py": "6b1fe398001808da42944d7460debb6fc0fad141628aba5b28aebbd081d2b320",
    "verify.py": "e7051620e0352c968cd3080e8b1bbb2036c5d9c003951fc8368b64e6a4bcf57f",
    "delivery.py": "8713039a7868fac6b92e881ef2e70da8c1e2dfb0e63075492aceed8f5e3be83d",
    "fresh_worker.py": "7684fd1f00c93e32746ae902333baceb32055cf4603e455563570350c6a49171",
    "baseline.py": "0513655049b5d7c0e4e57b3339596e718779cd7bcbcc49bafd76e14e5bd9dad1",
    "legacy_models.py": "031c25ba5a60cf3e6e4c8247553ee951ff71742c846eb103246140929f6cdfba",
    "legacy_reference.py": "1a24f9f116170ea54bd3242264d5cd2a150c3429e039a50cb11c3112e52247c1",
    "frozen_v9_modeling.py": "c4d948bcf9e1aab5e667283eb82889af44bc57147a0c2a234af6322044b55487",
    "v5_contract.json": "3446412f01d536a31123142b984392216efd29094965500251386eac1814ab69",
    "city_map.json": "539387a0a9f5e143228714dde7679c2848a019a1cbbaa87129b0d4e4c092f7f9",
    "reference_manifest.json": "82417467e331d6e470fd1058c4ea4edb5a47f98021efc822176bd9a04b99bbd5",
    "baseline_registry.json": "53e7ddb7fbfbe08162aa97fac8d5ad9dea4981aab3aa942b3e98ebdb2a3060b7",
    "PLAN.md": "7f47d2583fd60f7dfc1b0b776ab777894c56b253c89f3dcbbf06d1f216dc7092",
    "authorization.json": "e9d6503f1f182825cf6ac8cc985a3c0a99d84f4bb9c66deac8936eaf87b84e8e",
    "test_v22.py": "fc1ec8816414b13348de6d3d95be301cedcff7471b21dc7c3db230306c7196fc",
    "verify_results.py": "41dd0e06c44b58ec04ce4dd8e8e8c7eb2ff8ce9b25c66aff584bb752ff5ba8d7",
    "protocol_proposed.json": "e77375a0d6cc91d8cc3afdf9228a0775636ba42e11da53e1345556e60c540706"
  },
  "inputs": {
    "Train_dataset.csv": "ec40ebda74ba0bfbfe1af4417a4266a70eadf6e595500e99e5435e7bdc7ee51f",
    "Test_dataset.csv": "ee4137566adb1c0ce356a8c370e7b8a22f681bde492068efe11346d8d44b4f4d",
    "Data_Dictionary.csv": "2d86c00e27448593564a161197b6639ee1eb90c01fa523b67e7a09d2ef405c5e",
    "Sample_Solution.csv": "9519ba4d51eb2d59b8a71b054a62e1f4d48b72cf92930c31f5e5393a2cd135fb"
  },
  "model_contract_sha256": "d39629e11cda2c9192b175f6db2526e52d22896c3502d7c20b4511e478e503b0"
}
```

## Development

| Setting | Replacement weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|---:|---:|
| P35-SC38 | 25% | 19,641,684.48 | 19,642,696.88 | 7,349.07 | 7,208.58 | 0.03704% |
| P35-SC38 | 50% | 19,635,167.39 | 19,636,299.87 | 13,866.16 | 13,605.58 | 0.06990% |
| P35-SC38 | 100% | 19,624,632.36 | 19,625,943.63 | 24,401.19 | 23,961.83 | 0.12306% |

## Confirmation and decision

Retain V14: no development candidate qualified.
No development finalist qualified. Confirmation, bootstrap qualification, a new V22 CSV and a fresh V22 submission notebook were correctly skipped. Byte-identical scored V14 CSV/notebook are preserved as KEEP_V14 files.

Require positive gains on both development assignments and mean relative improvement >=0.20% against V14. One finalist is frozen before confirmation. Confirmation requires all three random gains positive, mean relative gain >=0.20%, a positive lower paired-row 95% gain bound (5,000 draws, seed 8319), and city/stratified deterioration <=1%. Labels have been historically examined; intervals are conditional robustness evidence, not an untouched test.

## Reproduction and compute

Conservative elapsed charge at packaging: 0.4572 hours under the four-hour cap. Free T4 only; no portal upload.
V7 uses the independently verified original-Mac fitted state; fresh Linux V7 retraining is not claimed. Qualified standalone notebooks fit every required TabPFN component afresh. Expected predictions are only verification witnesses after inference.
Fresh same-layout and CSV agreement require INR 0.01. Permutation, split-batch and individual probes permit at most INR 10 per prediction and INR 1 pilot RMSE change. No rounding is used to pass checks.

Original inputs, fold witnesses, source/environment/checkpoint hashes and all measured failures accompany this report. At most one manual V22 portal trial is recommended after complete qualification and reproduction. Keep V14 until an actual lower portal RMSE is returned. The one-time coding upload remains unconfirmed.

Sources: exact V14 computational identity, original forty-column feature definition and reviewed domain-adaptation research. See RESEARCH_SOURCES.md and PLAN_PROPOSED.md.
