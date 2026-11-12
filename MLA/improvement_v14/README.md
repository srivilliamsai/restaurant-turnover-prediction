# V14 backbone experiments

User-authorized eight-hour maximum, free T4 only. The frozen settings are I8 (TabICLv2) and P35_8 (actual TabPFN-3.5 Base), each with eight estimators, unchanged V9 features, raw INR-million targets and seed 10242.

Run `workflow.py STAGE --data-dir PATH --output-dir PATH`, in order: audit, pilot, screen, confirm, export. `run_colab.py --prepared` runs these stages in the owned CLI session `v14-free-t4`, backs up complete folds, collects outputs and releases the runtime. The pinned environment must be installed beforehand; the coordinator restarts the kernel before upload. Never run two coordinators against the same session.

The two blend routes and four weights create sixteen comparisons using only twenty new development component fits. One finalist may require twenty-five confirmation component fits. Failed pilots are engineering exclusions and do not constitute accuracy tests. Feasibility decisions are made without using development scores.

Same-layout repeat fits and fresh CSV reproduction require INR 0.01. The predeclared permutation/split/individual behavior tolerance is INR 10, with at most INR 1 RMSE change on the labeled pilot. Full-fold permutation and split probes plus twelve individual probes are mandatory. No rounding is used to satisfy checks.

All original targets remain in evaluation. IDs/targets cannot enter features, all vocabularies fit within training subsets, and cached controls require verified fold/row/source/runtime provenance. Checkpoints retain full context and output hashes. V7 uses the independently verified original-Mac fitted state; fresh identical Linux V7 retraining is not claimed.

`finalize.py` packages collected local evidence. `verify_results.py` independently reconstructs every blend and decision from complete folds; package again after verification. A qualified CSV additionally requires separate fresh notebook execution. Otherwise preserve byte-identical scored V9 files. No automatic portal upload exists.

The accepted contract is in PLAN.md. Runtime elapsed charges include setup and interruptions and preserve a two-hour delivery reserve. The deadline check records the live portal date and the unresolved timezone. Do not use validation gains as assumed portal gains or ranks.
