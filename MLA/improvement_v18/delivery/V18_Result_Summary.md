# V18 result

V18 completed. **Retain V14.**

V14 remains the best observed portal submission: **INR 11,736,110.7877709**. No V18 portal upload occurred.

| Model | Mean development RMSE | Portal RMSE |
|---|---:|---:|
| V14 control | 19,649,469.51 | 11,736,110.7877709 |
| Best V18 blend | 19,641,624.77 | Not submitted |

Best setting: **FT512 at 50% weight**. Mean local gain: **0.039920%**, versus the frozen **0.20%** requirement. Gains on assignments 42/71: INR -19,774.57 and 35,464.04.

| Recipe | Engineering result | Admission detail |
|---|---|---|
| FT512 | PASS | Passed the declared engineering checks. |
| EXA8 | engineering_excluded | No available kernel. Aborting execution. |
| FT1024 | PASS | Passed the declared engineering checks. |

30 complete outer-fold predictions and 10 fixed blends were independently reconstructed from hashed outputs. All 3,493 original labels are retained in the inputs; no labels were removed.

Each admitted setting passed its full-fold engineering, saved-state and fresh-fit checks. All engineering failures and budget exclusions are recorded in pilot.json with full pilot logs.

Confirmation and submission artifacts follow the frozen decision; no unqualified candidate was exported.

The scored V14 CSV and matching executed notebook are preserved byte-for-byte under KEEP_V14 filenames. The owned free T4 was released; account release evidence accompanies this result. The eighteen-hour ledger was not reset.
