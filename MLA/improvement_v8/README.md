# V8 restaurant-turnover research

Read REPORT.md first. All sixteen settings and sixty-four blend configurations failed the development gate; retain scored V7. The recommended_submission files are unchanged V7 backups, not a new CSV to upload.

V8_Research_Colab.ipynb is standalone and executed, with measured outputs. Its default fresh execution repeats the bounded screening and full V7 fit. Optional resumable checkpoints use strict hashes. Supply the four official CSVs; earlier project directories are not needed.

Local command: python improvement_v8/run_local.py all --data-dir /path/to/official/csvs
Stages: audit, screen, refine, confirm, export. --out-dir names the results directory; candidate/recommendation folders are created alongside it. --budget-hours defaults to 18, the maximum. No paid resource is required. No portal action was taken.

Refinement and confirmation were skipped intentionally because no setting qualified. The source repository was reviewed before training; competitor code and predictions are not embedded in the predictor. The 18-hour ceiling is not a requirement to repeat rejected experiments.
