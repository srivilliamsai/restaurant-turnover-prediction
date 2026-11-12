# V23 reproduction

Run from the original MLA project root. The original four CSVs and the V22 reference evidence are required for the independent historical audit; their expected hashes are recorded in history_audit.json.

1. `PYTHONPATH=.runtime_deps:improvement_v9/runtime_deps improvement_v9/colab_cli_env/bin/python improvement_v23/prepare.py` reconstructs the 35 matched P35 control contexts and exact partitions without fitting an estimator. It preserves the scored V14 fallback.
2. `improvement_v9/colab_cli_env/bin/python improvement_v23/download_checkpoint.py` downloads only public official weights and their license; no local dataset is sent by this command. The downloaded 1,625,774,719-byte checkpoint must match the published SHA256.
3. The current protocol, source_frozen.json and authorization.json are immutable execution inputs. Do not change the elapsed start or resurrect this expired one-hour run. A later experiment needs its own explicitly scoped protocol and clock.
4. Authorized Colab execution uses transport.py with the existing signed-in account; no credential or authenticated URL is stored in the delivery. Uploading data requires the user's explicit authorization recorded in authorization.json. Only the session named in the runtime receipts may be released.
5. `PYTHONPATH=.runtime_deps improvement_v9/colab_cli_env/bin/python improvement_v23/finalize.py` reconstructs complete saved development comparisons and checks original input, reference and source identities. Run after collecting remote results into results/ and resolved.json. This is verification, not model training.

The initial MLX attempt is preserved in engineering_history. It was a compatibility/feasibility attempt, not an accuracy measurement. Its unmerged public backend sources and converted weights differ from the final official Torch CUDA execution path. Public weights and runtime dependency binaries are deliberately excluded from the compact delivery archive.

The matching scored V14 CSV/notebook in delivery/ remain the fallback. An independent review that says PASS concerns artifact integrity and reconstructed evidence; only an explicit qualified decision concerns validation. Neither establishes a lower portal score.
