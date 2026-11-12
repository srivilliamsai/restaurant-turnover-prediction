# V19 execution progress

Original authorized clock: 2026-10-29 14:54:22 IST. Training stops by 21:24 IST; absolute delivery ceiling 22:54 IST. Free T4 only. V14 remains the verified best submission. No portal upload authorized.

Historical audit passed: 110 cached folds and 40 V15–18 comparisons. Original Mac control reproduction passed (max difference INR 1.49e-8); initial GPU BASE/P35 control pilots reproduced exactly. Nine repaired local safeguard tests and repaired data/reference audit passed.

First runtime: D8 passed engineering and generated ten development folds plus automatic screening. Metrics were not inspected before repairs. C8 failed query layout stability; Kumo hit old Numba versus NumPy 2.5. Engineering repairs fix C8 query block size with training-row padding, and pin candidate-only Numba 0.68.0 / llvmlite 0.50.0. No weights, labels, features, seeds, blend grid or numerical gates changed.

First VM became unavailable before its complete archive could be downloaded. Initial source bundle and five completed fold backups are preserved in engineering_repair_evidence. engineering_amendment.json records this limitation. Do not claim all initial artifacts were recovered.

Second allocation on the same selected account succeeded. Repaired setup completed; active import probes confirm baseline Torch 2.11 / NumPy 2.5.4 and candidate Torch 2.13 / NumPy 2.5.4 / Numba 0.68.0 on Tesla T4. Repaired source freeze completed before repaired development. run_colab.py --prepared is running with periodic local backups. Original elapsed-time budget retained.

Pending: repaired model pilots, budget-based admission, frozen development comparisons, one finalist confirmation if eligible, fresh notebook/export if qualified, independent reconstruction, packaging and owned-runtime release. No positive accuracy result confirmed yet.

## Run outcome

Development completed: 20 candidate folds and 15 fixed blend comparisons. Causilo at 25% weight improved both assignments with mean relative gain 0.013192422%, below the fixed 0.20% floor. TabDPT and the equal blend did not improve mean RMSE. No candidate advanced to confirmation and no new V19 submission was exported. Scored V14 fallback artifacts were exported byte-identically. Collected artifacts are local; owned free T4 release command succeeded. Independent reconstruction, archive verification and fresh account release check finish the delivery.

Final independent reconstruction passed. Fresh selected-account verification found zero active assignments after owned-runtime release. No V19 portal upload occurred.
