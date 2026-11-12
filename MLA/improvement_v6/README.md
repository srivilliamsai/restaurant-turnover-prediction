# Improvement 6

Read REPORT.md first: it states whether a new candidate passed or v5 is retained.
The executed notebook is standalone and accepts the four official competition CSVs.
In Colab choose CPU, run all, then download the CSV and executed notebook.

Local reproduction requires the versions recorded in results/audit.json. The launcher
selects the bundled desktop Python 3.12 when present and configures the existing
workspace dependencies and OpenMP library without changing the system.

```sh
python3 improvement_v6/run_local.py --data-dir /path/to/competition/data
```

Alternatively, run these stages with your compatible configured Python environment:

```sh
python prepare_reference.py
python workflow.py all --data-dir /path/to/competition/data
python deliver.py build
python execute_notebook.py --data-dir /path/to/competition/data
python deliver.py finalize
```

The default data directory for workflow.py and execute_notebook.py is the workspace.
Existing matching prediction caches are hash-checked; stale caches raise an error.
The frozen lock prevents changing the shortlist when resuming. To run an independent
new research round, use a new output directory/version rather than replacing results.
The saved notebook uses no prediction caches and requires no helper Python files.
No portal upload is performed. Check portal scoring details, remaining submissions
and deadline timezone before any qualified trial. Keep the original v5 files.
