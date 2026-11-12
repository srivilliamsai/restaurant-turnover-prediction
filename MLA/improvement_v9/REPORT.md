# V9 status — CPU complete; manual GPU run pending

**Keep scored V7: portal RMSE 12,024,316.5355221.** No new CSV qualifies from the CPU experiments. The four TabPFN settings have not trained; they await your Colab execution.

## Measured CPU results

Completed four EBM settings, forty outer-fold fits and sixteen blend configurations on five-fold assignments42/71. None improved V7 on both assignments. Best blend: `ebm_sqrt_i10`, weight 25%; gain on42 **INR 767.07**, gain on71 **INR -10,260.81**. Positive means lower RMSE. This mixed result fails the agreed gate.

No refinement, confirmation or bootstrap interval was fabricated. All3,493 original targets remain in the primary metric. Saved diagnostics cover all96 segment comparisons and160 fold comparisons. These validation scores are not portal scores.

## Verification

A fresh process executed all10 code cells of the standalone notebook in CPU mode, refitted forty EBM outer folds without V9 prediction checkpoints, and reproduced all32 development scores with maximum difference INR 0.000000000000. Hash-verified historical V7 OOF snapshots were explicitly reused as controls. Fresh full V7 refitting reproduced its scored predictions within INR 0.000000007451. The scored CSV and matching notebook remain byte-identical.

Cache reuse/stale/corrupt/interrupted handling, feature schemas, fold isolation, original fit-label retention, unseen/missing values, ID/target invariance, permutation and twelve malformed CSV cases passed. GPU execution is not covered by this CPU verification. The pinned TabPFN9.2.0 API and official open v2 checkpoint identity were statically checked, but no checkpoint has been downloaded or fitted here.

## Colab handoff

The Lite notebook fixes a browser-editor issue: the old embedded payload occupied one580,616-character line. Compression and line wrapping reduce the file from656,480 to481,311 bytes; all code lines are now at most157 characters. Embedded model/feature/fold sources are unchanged. New transport hashes and fresh setup/audit/feature checks passed. GPU execution remains pending.

A free Tesla T4 connected and pinned packages installed successfully. The automated files.upload transfer failed, and you requested manual upload. Follow COLAB_INSTRUCTIONS.md using V9_Research_Colab_Lite.ipynb and the four CSVs from V9_Official_Inputs.zip. The notebook defaults to trying TabPFN on CUDA, checks full-fold feasibility/reproducibility, and reports failures explicitly. It downloads V9_research_results.zip. Send that ZIP back for review.

Only validation-qualified candidates advance. Any qualified CSV gets a matching standalone coding notebook and separate-process reproduction check before the results archive is delivered. No Olympus upload or one-time coding submission was performed. First place is not established or promised.

## Sources and folder review

The downloaded author folder has the same parsed official data values. Its S08 manifest reports12,088,422.07, worse than V7; the CSV bytes do not match that manifest hash. Its recovery search combines components from different fold assignments; separate confirmation outputs are incomplete. These saved reports were not adopted as our measurements or model predictions.

Model references: [TabPFN documentation](https://docs.priorlabs.ai/models), [official v2 checkpoint](https://huggingface.co/Prior-Labs/TabPFN-v2-reg), [EBM API](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html). TabPFN attribution: Prior Labs; Hollmann et al., Nature2025, doi:10.1038/s41586-024-08328-6.
