# Reproducing V22 evidence

V22 changes only the original V14 P35 component's feature input: remove Service and Comfortablility, retaining 38 columns. Checkpoint, eight estimators, seed 10242, float32, predictive mean, target units and fixed 500-row adapter follow the exact V14 contract. `PLAN.md` is the accepted plan; `PLAN_PROPOSED.md` preserves the original proposal. `authorization.json` retains the original clock. The user-requested account change and interrupted engineering evidence are under `account_change_evidence/`.

## Reconstruct the measured decision without training

In the original project, after final collection:

```sh
PYTHONPATH=.runtime_deps improvement_v9/colab_cli_env/bin/python improvement_v22/verify_results.py
```

This reads saved complete predictions, validates their hashes and exact input/source/fold/target contexts, reconstructs each declared blend and any confirmation, and checks the decision. It does not fit models, allocate compute or upload a submission. `delivery/independent_review.json` records the measured verification result when this step completes.

The final portable archive places the four original CSVs beside `improvement_v22/`, with `results/`, `reference/` and `provenance/` inside that directory. With a compatible Python environment containing the original NumPy, Pandas, scikit-learn, SciPy and CatBoost versions, run:

```sh
python improvement_v22/verify_results.py
```

`resolved_models.json` records the actual runtime and checkpoint parameters. `source_frozen.json` identifies computational files. `archive_verification.json`, written at final packaging, records full archive verification. No token store or model-vendor credential belongs in the archive.

## Prediction reproduction

Only a fully qualified candidate receives a freshly executed standalone submission notebook. It refits the required TabPFN components and checks all expected outputs only after inference. Original-Mac V7 fitted state is embedded and verified; identical fresh Linux V7 retraining is not claimed. Same-layout/fresh CSV agreement requires an absolute difference below INR 0.01, without relative tolerance.

If no candidate qualifies, `KEEP_V14_*.csv` and its matching notebook are byte-identical to the historical scored V14 artifacts. A new V22 submission is not created. See the measured report for the actual outcome; the absence of a new CSV is not evidence of an unperformed experiment.

The operational clock includes setup, account-change interruptions and recovery. The coordinator releases only its owned session. Fresh account-release evidence is checked separately from the local release command. Original execution ledgers and source identities are historical evidence and must not be reset to make an old run appear authorized again.
