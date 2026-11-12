# V9 manual Colab handoff

Read COLAB_INSTRUCTIONS.md. Upload V9_Research_Colab_Lite.ipynb, select free T4 GPU, Run all, and upload the four official CSVs. Return V9_research_results.zip. No earlier project directories are needed.

CPU EBM screening found no qualified candidate. TabPFN/GPU training is pending manual execution. Preserve scored V7 and its matching coding notebook. No portal upload was made.

Local stages: python improvement_v9/run_local.py STAGE --data-dir /path/to/csvs --out-dir /path/to/results --budget-hours 18
STAGE is audit, pilot, screen, refine, confirm, export or all. After qualified CLI exports, run candidate_notebook.py verify-all in the same pinned environment to generate and freshly execute matching notebooks. The Colab notebook performs that delivery step automatically.

V9 adds interpret-core0.7.8 in its own runtime_deps, preserving V7/V8 dependencies. The official TabPFN package9.2.0 is installed for GPU notebook execution. Model versionv2, repository revision and checkpoint SHA are recorded. Resumable folds reject changed sources, data, folds, training labels, core runtime and GPU execution contracts.

The ZIP omits installed libraries, earlier-source experiment caches, interrupted browser-transfer bundles and large repeated execution directories. The notebook embeds source, verified V7 OOF controls and scored backups. Input CSVs are supplied separately.
