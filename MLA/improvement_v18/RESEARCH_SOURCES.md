# V18 research register

Checked October 29, 2026, IST. This is a focused review of primary model sources and measured local history. No competition labels, files or source code were sent to a new remote service during planning. Public source text, one published source archive and checkpoint metadata were read; no model weights were downloaded and no runtime was allocated.

## Selected directions

| Resource | What was verified | Decision |
|---|---|---|
| [Prior Labs fine-tuning regressor](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_regressor.py) and [base lifecycle](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_base.py) | Supported context sizing, epoch selection, all-data refitting, estimator-count controls, CRPS/MSE losses and scheduler control. | Two fixed context-size recipes; no private training-loop edits. |
| Published [TabPFN 9.2.0 release](https://pypi.org/project/tabpfn/9.2.0/) | Its regressor source SHA256 matches the source recorded by the executed V17 pilot: `315413cd044720d1edc9962d6697ed55cea90f3e7181d1a2a1c41ac8ccde18fa`. The published base source computes scheduler length from requested epochs. | Explicit constant learning rate in both selection and refit. This is a new recipe choice; no V18 accuracy result is inferred. |
| [Official fine-tuning example](https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_regressor.py) | General example and hardware recommendation, rather than a restaurant-specific recipe. | Local V17 feasibility is more directly relevant to FT512. FT1024 still needs a real-sized pilot. |
| [EXAONE author repository](https://github.com/LGAI-Research/EXAONE-Tabular) and [model card](https://huggingface.co/LG-AI-Research/EXAONE-Tabular) | Public regression checkpoint and approximately 21.1M-parameter model. | One new regression-prior experiment; author benchmark claims are not expected competition gains. |
| EXAONE [pinned config](https://github.com/LGAI-Research/EXAONE-Tabular/blob/d149d0f0441053f0cf6b2b45f4b2067a81619cf9/src/exaonetabular/config.py), [regressor](https://github.com/LGAI-Research/EXAONE-Tabular/blob/d149d0f0441053f0cf6b2b45f4b2067a81619cf9/src/exaonetabular/regressor.py), [presets](https://github.com/LGAI-Research/EXAONE-Tabular/blob/d149d0f0441053f0cf6b2b45f4b2067a81619cf9/src/exaonetabular/presets.py) | Current revision allows mean readout and uniform weighting; stores fitted preprocessing and accepts a categorical mask. Support limits and dtype are explicit. Older cached web content described different defaults. | Freeze explicit mean/uniform/float32/no-SVD configuration rather than inherit defaults. Verify trained-state inference and full-row use in execution. |
| [Public Hub metadata](https://huggingface.co/api/models/LG-AI-Research/EXAONE-Tabular?blobs=true) | Revision `093ad1c2613e1e95d26df8610451d2e02ed53cbe` lists the 84,502,160-byte regression artifact and LFS SHA256 `24891e14e649c6509f07ee62b5583e06544b68eaf9a9195ee58777eb9cf6fa0d`. | Metadata availability passed. Download integrity and T4 feasibility have not passed. |

Snapshots from author GitHub repositories are identified in [source_review_registry.json](source_review_registry.json). Published-release and checkpoint metadata are in [release_review_registry.json](release_review_registry.json). Public source download used exact commit URLs. The web reader could not fetch those commit URLs from its cache, so the local snapshots and hashes supply the exact reviewed text; do not substitute its stale default-branch excerpts.

Exact source revisions reviewed:

- EXAONE: `d149d0f0441053f0cf6b2b45f4b2067a81619cf9`.
- TabPFN current source: `1c0a4574aa3a9fa9f4f45612331a1fb08420e256`; planned runtime remains the separately reviewed published 9.2.0 release.
- TabFM: `fbb665569425fd2f490c6576b3af967876fe11ff`.

## Directions deferred for this finite experiment

[Google TabFM](https://github.com/google-research/tabfm) remains a credible reserve. Its current README specifies a separate newer backend environment and a default context-row limit of 100. Adding it now would need another explicit context policy, full-row coverage and engineering investigation. That is a scope decision, not a claim that it cannot run on a T4.

[Nori](https://github.com/Synthefy/synthefy-nori) was measured locally in V17 and did not meet the gain floor. Its high residual correlation and sensitivity results give a reason to deprioritize additional blend weights. This does not establish that every Nori variant would fail.

[Xiaomi-TabLDM](https://github.com/xiaomi-research/xiaomi-tabldm) failed the actual V17 permutation gate. No supported remedy was established in this planning turn, so the unchanged recipe is excluded rather than assigned another pilot budget.

V16's Mitra lifecycle exclusion, V15's small refinements and prior target-cap, square-root, feature-cleanup, TabM, RealMLP and estimator-scaling results remain historical evidence. This review does not rebrand them as new V18 hypotheses. Their broader inventory remains in [the V17 research register](../improvement_v17/RESEARCH_SOURCES.md).

## Evidence and limits

The [V18 read-only audit](review_v17.py) checks V17 predictions without importing its training code. It verifies twenty original fold files and reproduces five scores, including Nori's 0.096332% mean gain. It also preserves the specific fine-tuning pilot's timing and engineering evidence. These are measured facts; transferring them to changed learning rates or context sizes is only a starting forecast.

New random confirmation assignments improve partition coverage. They do not erase historical exposure to these labels or guarantee portal generalization. No external target matches, public competition predictions, leaderboard-based row adjustments or guaranteed RMSE/rank are part of V18.
