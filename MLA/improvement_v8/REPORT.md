# V8 result — retain scored V7

**Decision:** no new V8 candidate qualifies. Keep V7 Trial 1: user-reported portal RMSE **12,024,316.5355221**, rank 24 in the supplied screenshot. No V8 CSV should be uploaded.

## Measured comparisons

Completed all **16 settings**, **160 V8 outer-fold component fits**, and **64 blends** on identical five-fold assignments 42/71. Recomputed 128 original-scale development scores. No blend improved V7 on either assignment. Positive gain would mean improvement; the gains below are negative.

| Family | Best component | Weight | Gain on 42 (INR) | Gain on 71 (INR) |
| --- | --- | ---: | ---: | ---: |
| encoded | encoded_sqrt_200 | 25% | -3,134.43 | -8,518.56 |
| ridge | ridge_sqrt_10_base | 25% | -2,813.25 | -14,098.24 |
| native | native_sqrt_7 | 25% | -9,231.17 | -14,779.12 |

The closest approach, target-encoded square-root HGB with smoothing 200 at 25% weight, worsened RMSE by about INR 3,134 and 8,519. The article-inspired square-root Ridge hypothesis also failed promotion. Public-repository scores were not substituted for our measurements.

## Gates and limits

No setting passed development. Refinement and fresh confirmation assignments were **not run**, as specified by the plan. No confirmation confidence intervals were calculated. The fixed screen finished before the 18-hour ceiling; the experiment list was kept unchanged.

All 3,493 original target values remain in evaluation. IDs and targets are excluded from predictors. No external data, public predictions, copied model weights or leaderboard-derived row adjustments enter the models. Every assignment reuses historically examined labels; development results are not untouched external evidence or portal scores.

## Reproduction

- Fresh full V7 refit matched the scored CSV within **INR 0.000000007451**, below INR .01.
- All 9 code cells of the standalone notebook executed in a fresh process and isolated directory with no earlier project directories or V8 component checkpoints. It repeated all V8 fits and the full V7 refit; no saved errors. Hash-verified prior V7 OOF snapshots were explicitly reused for development baselines.
- All 128 notebook scores matched the independent workflow within INR .01; maximum difference **INR 0.000000000000**.
- Target-encoding own-label exclusion, training-only frequency maps, categorical limits, missing/unseen values, permutation, target/ID invariance and stale/corrupt/interrupted checkpoints passed.
- All 384 turnover-bin diagnostics and 640 fold comparisons are saved in results/development_segments.csv and results/development_folds.csv. No targets were filtered from the primary metric.
- The 500-row fallback CSV and matching notebook are byte-identical to scored V7. Original IDs/order, exact headers, finite nonnegative values and twelve malformed-file rejections passed.
- Ran locally on CPU. Colab compatibility is provided; no cloud execution or paid compute is claimed. No portal or one-time coding upload was performed.

## Repository review before training

Reviewed revision `2ce8c0657b788198a2d8b06066b351320881ac0d` of the author's repository. Square-root Ridge is documented. Its saved manifests report S05 at 12.115M and the leaderboard-adjusted S08 at 12.088M; these are author-reported records, not independently reproduced portal measurements. The article's later 2.893M result is not a demonstrated training recipe in this review and does not identify your current leader's method.

Sources: [Ridge source](https://github.com/RoloMagnus/restaurant-turnover/blob/master/src/winner_replication_study.py), [saved model manifest](https://github.com/RoloMagnus/restaurant-turnover/blob/master/experiments/champion_v2_manifest.json), [article](https://medium.com/@rolandolbouloyfascitelli/how-playing-volleyball-with-ai-agents-helped-me-predict-restaurant-turnover-and-win-a-hackathon-a3c696d6e900).

## Files to use

Keep the existing V7 CSV. `recommended_submission` contains the same CSV and matching coding notebook for convenience. `V8_Research_Colab.ipynb` reproduces the research; it is not a new winning submission. Preserve the one-time coding upload until your final choice. The best observed portal score remains **12,024,316.5355221**.
