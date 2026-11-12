# V13 turnover experiments

Run `workflow.py STAGE --data-dir PATH --output-dir PATH` in the recorded environment. Stages are audit, pilot, screen, confirm, export, in that order. The four original CSV inputs are required. The default output is `results/`.

`initialize.py` prepares immutable V9 controls and sources from the existing workspace without training. `test_v13.py` tests target treatments, second-moment units, selection, budget reserve and invalid caches. The audit independently verifies all 35 saved TabPFN baseline folds and their original-platform V7 controls.

Only a free Tesla T4 with the pinned TabPFN v2 checkpoint is supported for training. Dependencies, checkpoint hash, full parameters and distribution implementation hash are saved in `resolved_models.json`. No paid fallback or automatic portal upload exists.

The installed Colab CLI transport uses the owned session `v13-free-t4`. `run_colab.py` installs pinned packages, restarts a clean kernel, uploads the frozen bundle, executes all five stages in the foreground, backs up complete folds locally and releases that session in its cleanup handler. `--prepared` skips installation only. Session allocation is a separate explicit free-T4 CLI operation. The source identity in `source_frozen.json` must pass before upload; numerical source changes invalidate the existing run.

After collection and runtime release, `finalize.py` verifies and packages local artifacts. `verify_results.py` independently recomputes every measured blend and decision from the complete fold checkpoints. Run `finalize.py` once more after that review to include its evidence in the final ZIP. Neither finalization nor review fits models or uploads to the portal.

The eight-hour wall-time budget survives interruptions and includes setup/recovery; stage limits and a two-hour delivery reserve are enforced. Complete fold checkpoints contain treatment thresholds, training/validation identities, original training-target hashes, source hashes and runtime identity. Changed or partial checkpoints are rejected.

Completed output is `V13_Delivery.zip`, with measured report, all comparisons, provenance and the selected artifacts. If no candidate qualifies, the delivery explicitly retains the original V9 CSV/notebook. A new CSV is recommended only after frozen confirmation and separate notebook execution reproduce it within INR 0.01.

All turnover labels are retained for evaluation. Q99/Q995 are training sensitivity experiments, not declarations that high turnover values are wrong. SQ2 obtains the model-implied expected squared square-root target rather than squaring its expected square root. No first- or second-place score is assumed.
