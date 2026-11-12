# V23 result

Retain V14. V23 did not pass engineering admission; no qualified replacement was produced.

Best observed portal RMSE: **11,736,110.7877709 (V14)**. Latest recorded rank is 11 in the October 29 19:44:46 screenshot; current rank is unknown. No portal upload occurred.

## Experiment

One fixed L2-TF2 component: official LimiX-2 checkpoint, original 40 features, training-only preprocessing, two fixed transformation members, seed 10242. One blend: 75% V14 plus 25% LimiX. All 3,493 labels retained. Local MLX backend from pinned open PR 60; no claim of exact PyTorch equivalence.

The user shortened the available time to one hour before predictions. Initial and amended plans remain preserved. Data stayed on this Mac after automatic approval review rejected the proposed Colab upload. The allocated unused free T4 was released, and the release check found zero active assignments. No dataset upload occurred.

Independent audit: 35 original matched P35 contexts verified; 0 complete new folds; 0 development comparison(s) reconstructed. Byte-identical scored V14 CSV and notebook are included.

## Measured limitation

Stage `coordinator`: `AssertionError` — V23 pilot failed; inspect measured evidence

## Measured limitation

Stage `pilot`: `RuntimeError` — [metal::malloc] Attempting to allocate 11233757568 bytes which is greater than the maximum allowed buffer size of 4294967296 bytes.

## Recommendation

Use the exact scored V14 artifacts unless a completely validated, reproduced candidate subsequently returns a lower actual portal score. Local percentages do not establish a new rank. Failed engineering is not evidence that LimiX is less accurate. Do not spend the remaining deadline on an unplanned model or weight search.

Research, pinned source/checkpoint manifests, the executable adapter, measured logs and independent review are included. Public weights and installed dependency binaries are excluded from the compact archive. Downloading weights is reproducible from download_checkpoint.py; no private data are needed by that download.

Prepared 2026-10-29T22:48:31.473404+05:30.
