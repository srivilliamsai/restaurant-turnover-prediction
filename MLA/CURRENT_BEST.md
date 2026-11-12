# Current best observed submission

**V14 — portal RMSE 11,736,110.7877709; rank 9 in the user-supplied October 28 screenshot.** Rank is historical, not a live leaderboard query.

Previous best V9: 11,904,515.9864996. Reduction: 168,405.1987287 (1.41463%). V14 is 25% original V9 plus 75% TabPFN-3.5 Base (eight estimators).

- [Scored CSV](improvement_v16/reference/scored_v14.csv)
- [Matching executed notebook](improvement_v16/reference/scored_v14.ipynb)
- [Score record and attribution](improvement_v16/reference/v14_portal_trial.json)
- [Latest completed experiment result](improvement_v22/delivery/V22_Result_Summary.md)
- [V17 research proposal](improvement_v17/PLAN_PROPOSED.md)
- [V18 proposed extended experiment](improvement_v18/PLAN_PROPOSED.md)
- [V19 official-source research and proposed experiment](improvement_v19/PLAN.md) — execution complete; retain V14.

CSV SHA256: `39d49f79beb8549d04ffc78d134aad2407dc5e73914dc3c30e8bb162a3c0ba05`.

Notebook SHA256: `86fae4ea97d6fd35c2ed485c60f26f02c98fccb3cbeda9a10228cc48e788318e`.

The user performed the V14 upload. The score record attributes it to these exact local files; the screenshot itself does not show a file hash. V15 retry, V16 and V17 produced no qualified finalist and no new portal trial. V17 completed ten Nori 1.5 development folds and five fixed blends; the best was 50% Nori plus 50% V14, with mean relative local gain 0.096332%, below the frozen 0.20% requirement. Xiaomi failed inference stability; TabPFN fine-tuning passed engineering but exceeded the conservative validation workload budget. Independent reconstruction passed. The owned free T4 was released, and the selected account reported zero active assignments. Keep the one-time coding upload tied to the final model choice and actual portal state.

V18 completed twenty candidate development folds, ten matched unadapted control folds and ten fixed blend comparisons. FT512 at 50% weight achieved the highest mean relative local gain, 0.039920%, below the frozen 0.20% requirement and with a loss on assignment 42. FT1024 also failed development qualification. EXAONE Tabular was excluded before accuracy testing because its forced Flash Attention kernel was unsupported on the owned free T4. No candidate advanced to confirmation; no V18 submission was exported or uploaded. Keep the exact scored V14 artifacts. The run retained its original eighteen-hour cap and stopped after the declared candidates failed, without spending the unused allowance on unplanned tuning.

V19 completed twenty candidate development folds and fifteen fixed blends. The best blend, 75% V14 plus 25% Causilo, improved both development assignments with mean relative gain 0.013192%, below the unchanged 0.20% qualification floor. TabDPT and the equal blend did not improve mean RMSE. Both Kumo sizes were excluded by a dependency import failure. No candidate advanced to confirmation; no new V19 submission was exported or uploaded. Independent reconstruction passed, the owned free T4 was released, and the selected account had zero active assignments. The initial interrupted attempt and engineering repairs are documented without resetting the eight-hour elapsed ledger.

V20 completed 30 candidate development folds and 20 fixed blends. Best mean relative local gain: 0.002639% (I16 at 10%). Retain V14: no development candidate qualified. No V20 portal upload occurred. Independent reconstruction passed and the owned free T4 was released.

V21 completed both TabPFN-3.5 feature pilots, 20 candidate development folds and six fixed replacements of the original P35 share. Best S at 25% replacement weight gained 0.004861% on average, below the unchanged 0.20% floor, and lost assignment 71. C+S+G lost both development assignments at every weight. No finalist advanced to confirmation, and no new submission or portal upload occurred. Independent reconstruction of all six blends and 20 folds passed; exact scored V14 artifacts remain preserved. The owned free T4 was released and a fresh selected-account check found zero active assignments. Retain V14 and close this feature/model-search campaign.

V22 completed one exact P35-SC38 pair-removal setting, ten development folds and three fixed replacements. Best mean relative development gain: 0.123064%. Retain V14: no development candidate qualified. All three comparisons were independently reconstructed. No portal upload occurred; original V14 artifacts remain the observed best. The owned T4 was released; final selected-account verification found zero active assignments. The requested account change and repeated engineering retained the original four-hour clock. See improvement_v22/delivery/V22_Result_Summary.md.
