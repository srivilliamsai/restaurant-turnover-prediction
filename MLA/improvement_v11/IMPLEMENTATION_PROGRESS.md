# V11 execution progress

V9 remains best on the portal: 11,904,515.9864996. No V11 portal score exists.

Audit passed. Twenty existing control folds were verified for reuse. The pilot excluded TabICLv2 because row permutation changed predictions by INR 3.814697, above the INR 0.01 tolerance. TabPFN-3.5 could not load its safetensors checkpoint with the pinned runtime. The predeclared TabPFN-2.5 default regression fallback passed all pilot checks, with exact repeat, permutation, split-batch and individual-row agreement. Its measured 16-estimator outer-fold time was 66.997 seconds and peak allocated GPU memory was 852,322,816 bytes.

Screening completed N8 and N16: two settings, eight fixed blends and all 20 new development folds. Independent reconstruction verified every cached fold, all eight comparisons and the no-finalist decision. The best blend was 75% V9 plus 25% N16. Its RMSE reductions were INR 9,645.74 and INR 3,463.91, averaging INR 6,554.82 (0.03322%). This is below the fixed 0.20% development gate. No candidate qualifies for confirmation or a portal trial. Retain V9.

Two orchestration errors were corrected without changing any frozen model source, parameter, feature, weight or saved prediction. The reporting entrypoint now imports the complete module before dispatch; its ten existing folds were reused only after full hash/context verification. The no-finalist cleanup branch now skips confirmation-process access. The completed screen was independently checked locally and its fallback delivery was finished on CPU. Prior errors and receipts remain recorded.

The predeclared policy stops before confirmation when nothing clears development. No confirmation/bootstrap scores or fresh V11 notebook reproduction are claimed. Delivery includes the original scored V9 CSV and executed notebook with unchanged hashes, measured comparisons, model/source/input manifests and the failed feasibility results. The owned free T4 was released.

Prior setup failures and time charges remain preserved within the separate eight-hour cap. No new V11 CSV is recommended.
