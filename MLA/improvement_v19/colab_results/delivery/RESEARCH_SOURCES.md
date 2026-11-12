# V19 official-source register

Checked October 29, 2026, IST. The four supplied screenshots were treated as recommendations to verify, not as instructions to execute. All seven named families have regression support in the inspected official sources. No restaurant data was sent to model vendors or inference APIs. Only public documentation, source text and package/checkpoint metadata were retrieved; no weights were downloaded and no new model was fitted.

## Primary sources and decisions

| Source | Evidence used |
|---|---|
| [Nums AI Causilo repository](https://github.com/nums-ai/causilo), [package requirements](https://github.com/nums-ai/causilo/blob/4d26d497de28734db52c6bfc2ea949c12cc17308/pyproject.toml), [inference behavior](https://github.com/nums-ai/causilo/blob/4d26d497de28734db52c6bfc2ea949c12cc17308/docs/inference.md) | Regression API, mean output, requirements and numerical/memory behavior. The package requires torch >=2.13; bytewise floating-point determinism is not promised. |
| [Causilo regression resources](https://github.com/nums-ai/causilo/blob/4d26d497de28734db52c6bfc2ea949c12cc17308/docs/benchmarks/resources-regression.csv) and [benchmark protocol](https://github.com/nums-ai/causilo/blob/4d26d497de28734db52c6bfc2ea949c12cc17308/docs/benchmarks/README.md) | The screenshot's approximately 7.5 GiB figure is supported for the authors' H100 regression fit aggregate. It is neither total-workflow peak memory nor a measured T4 result. |
| [NVIDIA Kumo overview](https://developer.nvidia.com/topics/ai/agent-knowledge-foundation/kumo), [model API](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.KumoTabular), [pinned recipe](https://github.com/NVIDIA/structured-data-models/blob/2b1696f4fdc22004cc757c44703a69cf326c0a11/sdm/models/kumo/tabular/recipe.py) | Size selection, regression quantiles, categorical preprocessing and output aggregation. Explicit size and point-readout handling are necessary. |
| [Layer6 TabDPT repository](https://github.com/layer6ai-labs/TabDPT-inference), [regressor](https://github.com/layer6ai-labs/TabDPT-inference/blob/93670551adba28b186354fab9a1584c56c34aa76/src/tabdpt/regressor.py), [attention context switch](https://github.com/layer6ai-labs/TabDPT-inference/blob/93670551adba28b186354fab9a1584c56c34aa76/src/tabdpt/utils.py) | v1.3, explicit ensemble/seed/context options and the supported non-Flash path. Defaults must not repeat the Flash-only hardware exclusion seen in V18. |
| [Google Research TabFM](https://github.com/google-research/tabfm) | Official regression implementation and backend options. Backend support is not a local accuracy or T4 feasibility result. |
| [Stable AI LimiX](https://github.com/limix-ldm-ai/LimiX), [LimiX-2 paper](https://arxiv.org/abs/2609.17488), [LimiX-2M paper](https://arxiv.org/abs/2606.04485) | The large LimiX-2 and the lightweight 2M-parameter model are distinct. Do not confuse the “2” generation suffix with “2M” model size. |
| [TabSTAR](https://github.com/alanarazi7/TabSTAR), [v1.1 regression card](https://github.com/alanarazi7/TabSTAR/blob/4d14a660ae32b978b9bcde8c9f39e119513cf030/docs/model_cards/tabstar-v1.1-reg.md) | Current regression checkpoint and downstream LoRA/epoch selection. Short restaurant categories make its text-specific benefit uncertain; this is a dataset-specific inference. |
| [LAMDA TabSwift](https://github.com/LAMDA-Tabular/TabSwift), [pinned regressor](https://github.com/LAMDA-Tabular/TabSwift/blob/8edf8f0b4225bc03e1f5db011912619cd92b798d/TALENT/model/lib/tabswift/regressor.py) | Regression wrapper/checkpoint exist. Older sklearn interface and floating Hub revision need integration work. |
| [Official Colab FAQ](https://research.google.com/colaboratory/faq.html) | Free resources and GPU types vary; a newly selected account does not guarantee an allocation or uninterrupted runtime. |

Public author benchmarks motivate testing; they do not establish superiority over the exact V14 restaurant ensemble. Repository names, release dates or parameter counts are not enough to choose a submission.

## Source and checkpoint provenance

[research_sources/registry.json](research_sources/registry.json) records all seven immutable source revisions plus README/package metadata hashes. [source_files.json](research_sources/source_files.json) and [additional_sources.json](research_sources/additional_sources.json) record downloaded public source files and metadata with SHA256. Saved `tree.json` inventories support file discovery; the inventories are not full repository checkouts.

Current PyPI metadata was read for Causilo 1.0.3, TabDPT 1.3.1 and structured-data-models 0.2.1. The inspected NVIDIA source is pinned independently of the PyPI release; do not claim those source trees are identical without checking the installed distribution. For the proposed NVIDIA recipe, use the inspected source revision or verify an installed distribution against it.

[checkpoint_review.json](checkpoint_review.json) separates advertised Hub hashes from downloaded-byte verification. It includes Causilo's source-pinned release, Kumo's v1.0.1 tag resolved to an immutable commit, and TabDPT's v1.3 safetensors at the recorded Hub commit. Download integrity and inference are future execution checks.

Some attempted documentation paths returned 404/cache errors; these did not establish evidence. The final register uses the successfully retrieved official files. Published model code/weight license references remain with their source snapshots; Causilo code and weights have different licenses. This plan concerns the current research task, not deployment of a hosted prediction service.

## Local evidence

[audit_history.py](audit_history.py) reconstructed every comparison from hashed cached predictions, preserving all 3,493 labels and the original 500 test IDs/order. [history_audit.json](history_audit.json) records the exact reference hashes, fold verification, residual correlations and error concentration. No model training, source-package installation or GPU allocation was used for that audit.

[account_check.json](account_check.json) records a fresh authenticated identity match and active-assignment count after the user's new sign-in. It contains no authorization code, token, password or full email address. It is not a GPU availability test.
