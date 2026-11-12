# V15 result

V15 completed. Retain V14: the best experimental blend improved locally by **0.022838%**, below the required **0.20%**.

| Model | Mean development RMSE, seeds 42/71 | Actual portal RMSE |
|---|---:|---:|
| V14 control | 19,649,469.51 | 11,736,110.7877709 |
| Best V15 experiment | 19,644,981.91 | Not submitted |

The best setting was **N32-T100**, using replacement weight **50%**. It improved both development assignments, by INR 3,703.57 and INR 5,271.61; the mean RMSE reduction was INR 4,487.59. Its total blend was 25% V9, 37.5% original V14 TabPFN 3.5 and 37.5% N32-T100.

All five engineering pilots passed. All 50 development fold fits and 20 blend comparisons completed on the free T4. Independent reconstruction verified every comparison and the decision; maximum RMSE disagreement was INR 3.72529029846e-09. All 3,493 original labels were retained.

No candidate reached the frozen development gate. Confirmation, bootstrap qualification, a new V15 submission CSV, a fresh V15 submission notebook and a portal trial were skipped according to the plan. The scored V14 CSV and executed notebook remain byte-identical under KEEP_V14 filenames. V15 has no measured portal score or rank.

The owned T4 was released, and the final account check showed zero active assignments. Setup, account changes and interruptions remain charged to the original four-hour retry clock. The original eight-hour attempt, its ledger and its archive are preserved in prior_attempt; no ledger was reset.

The full table and methods are in V15_Report.md; screening.csv contains all 20 comparisons. Turnover-band, fold and largest-error sensitivity diagnostics, complete hashed folds, source/checkpoint identities and independent_review.json accompany this result. No sensitivity diagnostic changed selection. These historically examined labels provide conditional robustness evidence, not an untouched test.
