# V19 reproduction and result evidence

The measured decision is in results/decision.json after the export stage completes. Local validation and a portal score are different measurements; no V19 portal upload is performed by this workflow. The scored V14 CSV and notebook in reference/ remain the verified fallback.

## Audit without fitting

From the project root, run the project Python with PYTHONPATH containing .runtime_deps, improvement_v9/runtime_deps and improvement_v19. audit_history.py reconstructs the historical V14–V18 comparisons. verify_results.py reconstructs the completed V19 results, selection and any confirmation from hashed predictions; it requires the collected results and matching source_frozen.json. Neither verifier selects new recipes.

## Model execution

The workflow uses the four official CSV inputs with their recorded SHA256 hashes and all 3,493 labels. V14 is reconstructed as 0.25 times V9 plus 0.75 times its eight-member TabPFN-3.5 component. Its original V7 fitted state and Mac fold witnesses retain their verified platform boundary. New random confirmation assignments rebuild V7 controls on the original Mac and TabPFN controls on the owned T4.

New candidate code runs in a separate Python environment with pinned Torch 2.13. V14 controls retain Torch 2.11. runtime_lock.json, active_environment_probe.json, source_registry.json, checkpoint_registry.json and fitted metadata identify dependencies, source modules and actual checkpoint bytes. C8 uses fixed 256-row padded query blocks; Kumo regression averages its declared 999 quantiles; TabDPT uses the supported non-Flash attention path. Each admitted candidate must pass query behavior, actual saved-state reload, repeated fitting and full-data memory checks.

If one candidate passes all development and confirmation gates, delivery contains its submission CSV and a separately executed standalone notebook. The notebook freshly fits the required TabPFN controls and selected candidate while retaining the embedded verified original-Mac V7 state. It reads expected predictions only after inference. If no candidate qualifies, delivery contains byte-identical KEEP_V14 files; no new V19 submission CSV is exported.

## Engineering amendment and compute

engineering_amendment.json documents the first attempt and the repairs based only on Causilo query stability and Kumo import failures. The first VM became unavailable before its full archive could be downloaded. Five completed initial folds and the first source bundle were recovered in engineering_repair_evidence/. Initial accuracy scores were not inspected for repair choices. The repaired source freeze precedes repaired development.

The original eight-hour authorization includes setup, interruptions, duplicate fits and recovery. Training has a 6.5-hour ceiling and delivery reserves 1.5 hours. Free T4 only; the selected account and owned session are checked explicitly. runtime_release.json and a fresh account_release_check.json provide release evidence after completion.
