# V18 verification and reproduction

V18 tested the frozen FT512 and FT1024 recipes on both five-fold development assignments. Neither recipe qualified. EXA8 was excluded before accuracy testing because its forced Flash Attention kernel did not run on the owned T4. No V18 submission CSV or fresh submission notebook was produced. KEEP_V14 files are the original scored V14 artifacts.

The archive is an evidence bundle. Its `source/` directory contains the exact model implementation and verification scripts; `source_frozen.json` records model source hashes. `cache/` contains all 20 candidate and 10 matched-control outer-fold predictions and their metadata. `provenance/` and `reference/` retain the original control witnesses. `official_inputs/` holds all four unchanged competition inputs. `execution_logs/` and `execution_transport/` record the run and cleanup. Model weights and credentials are excluded; exact checkpoint identities are recorded in the registries.

In the original project workspace, independently reconstruct all ten blends and the no-finalist decision with:

```sh
PYTHONPATH=.runtime_deps:improvement_v9/runtime_deps:improvement_v18 improvement_v9/colab_cli_env/bin/python improvement_v18/verify_results.py
```

The compatible local numerical runtime is required. The verifier checks source and data hashes, all fold contexts and targets, original control witnesses, saved prediction hashes, all fixed weights, the selection decision and the preserved V14 files. This command reads saved predictions and does not train or use Colab.

The remote model environment and package versions are recorded in `resolved_models.json`; exact reviewed source and checkpoint revisions are in the registries. The original execution authorization and elapsed-time ledger are retained. A future fresh training run needs its own explicit authorization and execution clock; these receipts must not be reset to reuse the completed run's allowance.

The zip stores report files at its root and implementation files in `source/`. It is not a directly runnable project checkout. Restore the implementation to an `improvement_v18/` directory, place official inputs in its parent directory, and restore prediction caches to `improvement_v18/results/cache/` and result JSON/CSV files to `improvement_v18/results/` before using the original verification command in another workspace. Preserve all recorded model, source, fold and data identities.
