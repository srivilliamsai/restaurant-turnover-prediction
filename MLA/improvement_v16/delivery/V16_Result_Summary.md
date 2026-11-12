# V16 result

V16 completed. **Retain V14.**

V14 remains the best observed portal submission: **INR 11,736,110.7877709**. No V16 portal upload occurred.

| Model | Mean development RMSE | Portal RMSE |
|---|---:|---:|
| V14 control | 19,649,469.51 | 11,736,110.7877709 |
| Best V16 blend | 19,643,213.28 | Not submitted |

Best setting: **TM32-PLE at 10% weight**. Mean local gain: **0.031839%**, versus the frozen **0.20%** requirement. Gains on assignments 42/71: INR 3,140.62 and 9,371.83.

10 complete outer-fold predictions and 5 fixed blends were independently reconstructed from hashed outputs. All 3,493 original labels remain in evaluation.

TabM passed the full-fold engineering checks and exact fresh-fit repetition. The official Mitra-v2 adapter was excluded before accuracy testing because it removes its fitted-model directory and exposes no supported persistent-child prediction interface; the source revision and evidence are preserved.

No development finalist qualified; confirmation, bootstrap qualification, a new submission CSV and a fresh V16 submission notebook were skipped.

The scored V14 CSV and matching executed notebook are preserved byte-for-byte under KEEP_V14 filenames. The owned free T4 was released; account release evidence accompanies this result. The six-hour ledger was not reset.
