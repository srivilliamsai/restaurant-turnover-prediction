# V22 researched proposal

Read [PLAN.md](PLAN.md) for the proposed experiment and [RESEARCH_SOURCES.md](RESEARCH_SOURCES.md) for the primary-source review and alternatives.

Recommendation: test the exact V14 TabPFN-3.5 component with Service and Comfortablility removed; one 38-column component, three fixed blend weights, unchanged qualification gates. This extends an older tree-model ablation to the successful P35 backbone. It does not assume the survey fields are corrupt.

The planning audit found held-out train/test domain AUC around 0.69, falling near 0.50 without that pair. It verified ten original V14 P35 development-fold outputs and the scored CSV/notebook hashes. Missingness alone had weak domain signal. [Evidence](evidence_audit.json) includes full methods, metrics, historical comparison hashes and limitations.

Reproduce the CPU planning audit from the project root:

```sh
PYTHONPATH=.runtime_deps improvement_v9/colab_cli_env/bin/python improvement_v22/audit_evidence.py
```

Only train/test membership classifiers were fitted for the diagnostic. No V22 turnover model, GPU allocation, new submission or portal trial occurred. Best observed portal RMSE remains **11,736,110.7877709 (V14)**. Current best artifacts and previous experiment ledgers are unchanged.
