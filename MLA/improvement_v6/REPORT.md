# Improvement 6 — measured result

**Student:** submission · submission · **Team:** submission

**Decision: retain portal-scored v5.** No new candidate passed the predeclared gates. Keep the portal-scored v5 model. The delivered CSV is a byte-identical copy of v5; uploading it again would not improve the score.

## Score context

- User-confirmed v5 portal RMSE: **12,046,722.3691139**.
- Second place in the supplied screenshot: **11,588,949.5634295**; first: **4,796,740.5600053**.
- Required reductions from v5: **3.80%** and **60.18%**, respectively. No new portal score or rank is claimed.

## Completed experiments

All six predeclared settings were evaluated on both five-fold development assignments (42/71), with two model seeds. Each setting was tested alone and at 25%, 50%, 75% weight with v5. All validation targets remained unchanged and were scored in INR. The table shows the best development blend for each of the six settings.

| name | weight | gain_42 | gain_71 | mean_rmse |
| --- | --- | --- | --- | --- |
| sqrt_crossfit | 0.25 | 4,605.34 | -800.67 | 19,826,841.01 |
| domain_huber | 0.25 | -12,538.55 | -2,529.88 | 19,836,277.56 |
| native_l2_15 | 0.25 | -10,361.94 | -18,910.81 | 19,843,379.72 |
| native_l2_7 | 0.25 | -12,390.86 | -33,749.98 | 19,851,813.76 |
| native_huber_15 | 0.25 | -59,344.91 | -52,311.66 | 19,884,571.63 |
| native_huber_7 | 0.25 | -73,556.87 | -67,420.36 | 19,899,231.96 |

Positive gain means lower RMSE than v5. Full results, including all losing settings, are in `results/screening.csv`.

The frozen-v5 implementation was freshly refitted on both development assignments and reproduced the historical predictions exactly (maximum difference INR 0.00). Any reused confirmation baseline is checked against original input, source, contract and file hashes; assignment 18421 is additionally retrained in the delivered notebook.

## Frozen confirmation

No setting passed both development assignments; no finalist was advanced.

Acceptance requires positive gain on both random confirmation assignments and no more than 1% city-group RMSE deterioration. The development shortlist was locked before any new confirmation. For any city comparison, groups use the existing v5 clean alias map and v5 is refitted on exactly those same groups. The map remains approximate for unrecognized spellings.

The empty shortlist triggered the planned v5 fallback; no new confirmation or portal trial was warranted.


Turnover-bin RMSE and squared-error shares for all six settings are in `results/development_segments.csv`; fold and tail counts are in `results/development_folds.csv`. For any finalists, corresponding confirmation diagnostics are in `results/segments.csv` and `results/folds.csv`. No subset diagnostic replaces full RMSE.

## Correctness and reproducibility

- All four original CSV hashes match the earlier measured inputs.
- Data directory is explicit; defaults to the workspace. No Downloads path is required.
- Native LightGBM category vocabularies and early stopping are fitted inside training folds; early stopping uses original-scale-equivalent RMSE on targets scaled by one million.
- Square-root correction uses three inner OOF folds; outer validation labels are excluded.
- Distribution weights use cross-fitted train/test membership probabilities, without target or ID; density ratios are clipped to 0.5–2.0, then normalized. Supplied unlabeled test covariates are used only by this experiment.
- Fresh v5 full refit maximum difference from scored CSV: **INR 0.00000001**.
- Standalone notebook: **10 code cells executed in a new process**, one complete five-fold assignment freshly retrained without caches, and all selected full fits freshly retrained.
- Notebook full-fit maximum prediction difference: **INR 0.00000000**.
- Passed row permutation, missing values, unseen city, target/ID invariance, CSV round-trip, finite nonnegative predictions and 12 malformed-file rejections.
- Final CSV: **500 rows × 2 columns**, original restaurant IDs/order, 501 lines including header.

## Limits and submission handling

Every split reuses labels examined in previous rounds. Results are robustness evidence, not an untouched external test or an unbiased assessment of the complete historical search. Local RMSE must not be subtracted from the portal score to forecast a new score.

The portal scoring subset, remaining allowance and deadline timezone are unverified. **No portal uploads were performed.** The maximum planned allowance is two qualified trials. Preserve exact CSV hashes and returned portal scores in `portal_trials.csv`. No unconditional prediction-distance score bound is applied.

## Files

- `submission.csv` — recommended submission or preserved fallback, according to the decision above.
- `submission.ipynb` — standalone executed Colab-compatible notebook.
- `scored_v5_backup/` — exact scored v5 CSV and matching original notebook.
- `results/` — complete predictions, fold diagnostics, frozen lock and execution evidence.
- `delivery_manifest.json` — hashes, versions, decision and provenance.

CSV SHA256: `66a673efc5f31bc9f4010de033999fd3712444b8f6dd0478c97e77bb3934630d`

Primary implementation references: [LightGBM native categories](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html), [scikit-learn validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
