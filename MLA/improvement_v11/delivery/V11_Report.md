# V11 measured report

**Retain V9. No new V11 CSV is recommended.**

Best observed portal score remains V9: 11,904,515.9864996. No V11 portal score is inferred from validation. All 3,493 original targets and original test IDs/order are retained. IDs and validation labels are excluded from predictors.

Features are the unchanged 40-column V9 builder. Four predeclared model settings use raw targets in INR millions; feasibility may exclude a family before development metrics are read. Blends use (1-w)*V9+w*M with w 0.25/0.50/0.75/1.00.

## Feasibility

{
  "I": {
    "status": "unavailable",
    "reason": {
      "error_type": "AssertionError",
      "message": "permutation dependence INR 3.814697265625"
    }
  },
  "N": {
    "status": "PASS",
    "training_rows": 2794,
    "validation_rows": 699,
    "fold_seconds": 66.99661017600002,
    "repeat_difference_INR": 0.0,
    "behavior": {
      "permutation_difference_INR": 0.0,
      "label_id_difference_INR": 0.0,
      "batch_difference_INR": 0.0,
      "individual_difference_INR": 0.0,
      "missing_unseen_inputs": true
    },
    "peak_gpu_allocated_bytes": 852322816,
    "engineering_seconds": 416.527028071
  }
}

Engineering-only selected settings: N8, N16

## Screening

| Setting | Weight | Gain 42 | Gain 71 | Mean relative gain |
|---|---:|---:|---:|---:|
| N8 | 25% | 6,698.38 | 2,808.14 | 0.0241% |
| N8 | 50% | 3,161.00 | -5,095.55 | -0.0049% |
| N8 | 75% | -10,606.63 | -23,698.23 | -0.0869% |
| N8 | 100% | -34,583.13 | -52,969.72 | -0.2219% |
| N16 | 25% | 9,645.74 | 3,463.91 | 0.0332% |
| N16 | 50% | 8,942.13 | -3,818.64 | 0.0130% |
| N16 | 75% | -2,109.73 | -21,835.75 | -0.0607% |
| N16 | 100% | -23,492.46 | -50,558.09 | -0.1876% |

The best development blend was N16 at 25% weight: mean RMSE reduction INR 6,554.82 (0.03322%). It failed the predeclared 0.20% minimum. No finalist was frozen, so confirmation, bootstrap qualification, new CSV export and fresh V11 notebook execution were correctly skipped. No V11 portal trial is recommended. The preserved V9 CSV and notebook are byte-identical to the scored originals; this round makes no new numerical reproduction claim.

## Execution record

{
  "status": "COMPLETE_NO_FINALIST",
  "completed_new_development_folds": 20,
  "settings": 2,
  "blends_independently_recomputed": 8,
  "confirmation_performed": false,
  "confirmation_skip_reason": "No development blend meets the fixed 0.20% minimum mean relative gain.",
  "new_csv_exported": false,
  "fresh_v11_notebook_executed": false,
  "preserved_v9_hashes_verified": true,
  "model_sources_unchanged_from_pilot": true,
  "paid_resources_used": false,
  "portal_uploads": 0,
  "owned_colab_runtime_released": true,
  "orchestration_repairs": [
    "Load complete workflow module before invoking main so reporting helpers are defined; reuse ten complete folds only after hash/context checks.",
    "Skip confirmation process accesses when frozen finalist is None; complete fallback packaging locally without training."
  ],
  "engineering_failures_preserved": true,
  "runtime_release_evidence": "runtime_release_verification.json"
}

## Frozen confirmation

Choice: null


Development requires positive gains on both assignments and mean relative gain >=0.20%. Confirmation requires all three random gains positive, mean relative gain >=0.20%, a 5,000-resample paired-row lower 95% gain bound above zero, and city/stratified deterioration <=1%. Reused labels and conditional intervals do not constitute an untouched external test. All failures remain reported.

## Reproduction boundary

V7 uses the user-owned original-Mac fitted state verified against scored V7. Linux V7 retraining is not claimed to reproduce it. The unchanged V9 TabPFN v2 and any qualified new backbone train afresh in the standalone notebook. Expected predictions are read only after inference to test reproduction; they are not predictors.

Official pretrained checkpoints come from Prior Labs and the TabICLv2 maintainers (soda-inria/tabicl). Revisions, hashes, parameters and exact dependencies are recorded in resolved_models.json. No competition-trained public model or competitor predictions are used.

## Compute and submission

Recorded task charge: 0.5516 hours under the separate eight-hour V11 maximum. Free compute only. Earlier V10 accounting was not reset or extended. The final compute ledger is authoritative.

No automatic portal upload. After every gate and fresh reproduction passes, allow one user-uploaded trial and record its exact CSV hash and actual score. Retain the lowest observed score. Keep the one-time coding upload pending. The displayed October 29, 11:59 PM deadline timezone still needs verification. No first-place result is promised.
