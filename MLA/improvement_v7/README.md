# V7 restaurant turnover

Read REPORT.md before uploading a CSV. Files in candidates/trial_1 and trial_2
are distinct experimental candidates, each with its matching executed notebook.
The report states their local evidence and mathematical first-place feasibility.
The user's October 26 screenshot reports 12,024,316.5355221 and rank 24,
associated with Trial 1 from the conversation. Trial 1 is now the best reported
submission; preserve v5 and both matching notebooks. See portal_result_2026-10-26.json.
V7_Delivery.zip remains the original pre-upload delivery snapshot.

For Colab: upload a candidate notebook, select CPU, run all, then upload the four
official CSVs. To rerun all research use V7_Research_Colab.ipynb instead. Checkpoints
resume on the same runtime; enable the notebook's optional Google Drive setting
to keep checkpoints through runtime deletion. No credentials are embedded.

Local reproduction uses the compatible Python/dependency versions in results/audit.json:

```sh
python3 improvement_v7/run_local.py --data-dir /path/to/the/four/official/csvs
```

Or, with the compatible environment already configured, from improvement_v7:

```sh
python workflow.py all --data-dir /path/to/the/four/official/csvs
python deliver.py build
python execute_notebooks.py --data-dir /path/to/the/four/official/csvs
python deliver.py finalize
```

workflow.py defaults to the workspace data directory and can train references
from scratch without older improvement folders. Preserved v5 files and its minimal
hash manifest are included. Stale caches are rejected. Initial portability-check
artifacts are archived separately and are not used in the final results.
Submit only one final executed coding notebook after choosing the best portal CSV.
