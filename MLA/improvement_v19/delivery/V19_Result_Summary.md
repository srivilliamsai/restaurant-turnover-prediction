# V19 result

V19 completed. **Retain V14.**

V14 remains the best observed portal submission: **INR 11,736,110.7877709**. No V19 portal upload occurred.

| Model | Mean development RMSE | Portal RMSE |
|---|---:|---:|
| V14 control | 19,649,469.51 | 11,736,110.7877709 |
| Best V19 blend | 19,646,877.24 | Not submitted |

Best setting: **C8 at 25% weight**. Mean local gain: **0.013192%**, versus the frozen **0.20%** requirement. Gains on assignments 42/71: INR 1,694.55 and 3,489.97.

| Recipe | Engineering result | Admission detail |
|---|---|---|
| C8 | PASS | Passed the declared engineering checks. |
| KS8 | engineering_excluded | AttributeError: module 'numba.cuda.types' has no attribute 'NPDatetime' |
| D8 | PASS | Passed the declared engineering checks. |
| KM8 | engineering_excluded | AttributeError: module 'numba.cuda.types' has no attribute 'NPDatetime' |

20 complete outer-fold predictions and 15 fixed blends were independently reconstructed from hashed outputs. All 3,493 original labels are retained in the inputs; no labels were removed.

Each admitted setting passed its full-fold engineering, saved-state and fresh-fit checks. All engineering failures and budget exclusions are recorded in pilot.json with full pilot logs.

Confirmation and submission artifacts follow the frozen decision; no unqualified candidate was exported.

The scored V14 CSV and matching executed notebook remain preserved in reference/. The owned free T4 was released; account release evidence accompanies this result. The original eight-hour ledger was not reset.

An earlier runtime generated ten TabDPT development folds and automatic screening before an engineering repair. Those accuracy metrics were not inspected to choose repairs. Causilo query stability and a Kumo dependency import failure motivated the amendment. The first VM became unavailable before its complete archive could be downloaded; five completed fold backups and its source bundle were recovered. These interrupted and duplicate fits remain charged to the original elapsed-time ledger. See engineering_amendment.json and engineering_repair_evidence/.
