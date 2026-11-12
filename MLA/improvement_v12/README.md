# V12 workflow

Run `python run_stage.py STAGE --data-dir DATA --output-dir OUTPUT --budget-hours 18`, where STAGE is audit, pilot, screen, confirm or export. The full module loads before dispatch. Source and data identities are frozen in audit; numerical model parameters, environment and checkpoint are frozen in pilot.

The finite CLI coordinator is `transport/continue_finite.py`, launched through `transport/launch_detached.py`. It uses the existing authorized CLI OAuth store, an owned free T4 session named v12-free-t4, a clean kernel after installation, local complete-fold backups and absolute deadline/budget guards. Credentials stay outside artifacts. No paid compute or portal upload is performed.

Use `transport_run/status.json`, `transport_run/coordinator.log` and `colab_results/run.log` for live progress. The compute ledger is retained across resumes. Reuse requires complete data/metadata pairs with matching source/configuration/input/row/fold hashes.

After a Colab disconnection, individual pilot prediction checks also resume from `pilot_query_cache`. First-fit and fresh-repeat-fit predictions have separate identities. Raw query identities include IDs and labels so isolation checks cannot reuse their control prediction. Cached values are preserved exactly; no rounding or numerical tolerance changes. The original measured inference duration remains available for compute forecasting. A transport heartbeat that does not advance for five minutes stops the stale client and preserves completed work.

The user extended the cap from twelve to eighteen hours and required completion of all four settings. T32, R1 and R4 have passed; their exact numerical contracts and original witnesses are preserved. T4x16 must complete its interrupted check before screening, and all sixteen blends must finish before selection. A failed or interrupted check cannot be silently omitted. Earlier charges and the three-hour delivery reserve remain protected. No model, feature, blend or qualification rule changes.

Use `transport/launch_detached.py --resume-existing` only to reuse this workflow's existing free T4 after a controlled coordinator replacement. It preserves the installed numerical environment and verified individual pilot checkpoints; source/configuration hashes must still match. Normal fresh-session launch remains available after a real Colab disconnection.

V9's verified original-Mac fitted V7 state is the reproducible baseline boundary. Each newly selected component trains afresh in the executed standalone notebook. Expected predictions are read only after inference for checking. The delivered report explicitly distinguishes trained, skipped, failed and incomplete experiments.
