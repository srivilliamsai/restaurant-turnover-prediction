# October 18 restaurant turnover experiment

Student: submission · submission · Team submission

The final decision, scores and artifact links are in REPORT.md once execution completes.

## Reproduce the experiment locally

Use a Python environment with numpy, pandas, scikit-learn, scipy, catboost, matplotlib and nbformat. Source CSVs are read from the user Downloads folder by load_data.

1. `python new_models.py` screens 22 settings on split 42.
2. `python select_and_confirm.py select` runs the second development split and freezes a blend.
3. `python select_and_confirm.py confirm` runs the frozen comparison.
4. `python fit_candidate.py` fits all models and exports the CSV.
5. `python make_notebook.py` embeds the complete implementation in a standalone Colab notebook.
6. `python execute_notebook.py` executes all notebook cells, using explicitly verified fold caches and fresh full fits.
7. `python check_uncached_notebook.py` independently retrains a complete split from the embedded notebook code.
8. `python finalize.py` checks artifacts, writes the measured report, and promotes only if the predeclared validation rule passes.

For Google Colab, use the delivered notebook directly and upload the four official input CSVs. It needs no local helper scripts.

Local validation results are not hidden-test scores. No upload to Olympus is performed by these scripts.
