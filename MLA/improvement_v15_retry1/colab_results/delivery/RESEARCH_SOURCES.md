# V14 resource review

Checked October 28, 2026. Primary sources only. These describe available methods and engineering behavior; none demonstrates improvement on the supplied restaurant data.

| Source | Finding used in the plan |
|---|---|
| [Prior Labs model catalog](https://docs.priorlabs.ai/models) | Distinguishes local 3.5 Base from hosted Plus/Thinking. Select local Base for the proposed free-compute workflow. |
| [Official TabPFN-3.5 model card](https://huggingface.co/Prior-Labs/tabpfn_3_5) | Names the September 9 safetensors checkpoint, describes regression support and synthetic-only pretraining. |
| [Official v9.2.0 checkpoint loader](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/checkpoint.py) | Chooses safetensors versus torch loading by file suffix. Check the actual installed code and local artifact path when diagnosing V11. |
| [Prior Labs local weight access](https://docs.priorlabs.ai/models/accessing-model-weights) | Documents authentication, license acceptance, offline use and preservation of original weight filenames. Access is an engineering prerequisite; existing access has not been tested in this planning turn. |
| [Official TabICL repository](https://github.com/soda-inria/tabicl) and [API reference](https://tabicl.readthedocs.io/en/latest/api.html) | Provides the v2 regressor, explicit predictive-mean output, estimator count, precision and inference-memory settings. |
| [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) | Explains operation-order and batching differences in floating-point calculations. Supports investigating a small permutation discrepancy; does not prove V11's discrepancy was harmless. |
| [Official TabPFN regressor fine-tuning example](https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_regressor.py) | Recommends 80 GB CUDA hardware for its example. Exclude fine-tuning from the bounded main round rather than assume T4 feasibility. |
| [scikit-learn nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) | Explains model-selection optimism. Reused confirmation labels are not independent external evidence. |

Local evidence checked:

- `improvement_v10/colab_results/delivery/V10_Report.md`: measured portal regression after local qualification.
- `improvement_v11/delivery/resolved_models.json` and `pilot.json`: actual 2.5 fallback and TabICL's INR 3.814697265625 exclusion.
- `improvement_v12/colab_results/delivery/V12_Report.md`: estimator scaling and RealMLP outcomes.
- `improvement_v13/delivery/V13_Report.md`, `screening.csv`, `independent_review.json`, original inputs, baseline registry and complete candidate fold outputs.
- V6–V8 reports/plans: avoid repeating domain weighting, specialists, residual models and encoded Ridge/HGB searches.

Unresolved items are explicitly retained in the proposal: exact 3.5 loader failure cause, T4 feasibility, current weight access, new-model accuracy, fresh execution agreement and the portal deadline timezone. No cloud allocation, new model fit or portal upload was performed during this review.


V15 primary sources reviewed October 28, 2026:
- https://docs.priorlabs.ai/improving-performance
- https://docs.priorlabs.ai/improving-performance/model-parameters
- https://raw.githubusercontent.com/PriorLabs/TabPFN/v9.2.0/src/tabpfn/preprocessing/datetimes.py
- https://raw.githubusercontent.com/PriorLabs/TabPFN/v9.2.0/src/tabpfn/inference_tuning.py
These justify testing representations/temperature, not an assumed improvement.
