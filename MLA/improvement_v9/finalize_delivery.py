"""Package measured V9 evidence and the manual GPU execution handoff."""
import json
import shutil
import zipfile
from pathlib import Path
import pandas as pd
from engine import ROOT, Engine, sha, save


def main():
    execution=json.loads((ROOT/'results/research_notebook_execution.json').read_text())
    fresh=Path(execution['fresh_directory'])
    for p in (fresh/'results').rglob('*'):
        if p.is_file():
            target=ROOT/'results'/p.relative_to(fresh/'results');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    shutil.copytree(fresh/'recommended_submission',ROOT/'recommended_submission',dirs_exist_ok=True)
    import analyze
    analyze.run()
    e=Engine();table=pd.read_csv(e.out/'screening.csv');best=table.iloc[0]
    report=f'''# V9 status — CPU complete; manual GPU run pending

**Keep scored V7: portal RMSE 12,024,316.5355221.** No new CSV qualifies from the CPU experiments. The four TabPFN settings have not trained; they await your Colab execution.

## Measured CPU results

Completed four EBM settings, forty outer-fold fits and sixteen blend configurations on five-fold assignments42/71. None improved V7 on both assignments. Best blend: `{best['name']}`, weight {best['weight']:.0%}; gain on42 **INR {best['gain_42']:,.2f}**, gain on71 **INR {best['gain_71']:,.2f}**. Positive means lower RMSE. This mixed result fails the agreed gate.

No refinement, confirmation or bootstrap interval was fabricated. All3,493 original targets remain in the primary metric. Saved diagnostics cover all96 segment comparisons and160 fold comparisons. These validation scores are not portal scores.

## Verification

A fresh process executed all{execution['code_cells']} code cells of the standalone notebook in CPU mode, refitted forty EBM outer folds without V9 prediction checkpoints, and reproduced all32 development scores with maximum difference INR {execution['maximum_score_difference']:.12f}. Hash-verified historical V7 OOF snapshots were explicitly reused as controls. Fresh full V7 refitting reproduced its scored predictions within INR {execution['v7_reproduction']['maximum_difference_INR']:.12f}. The scored CSV and matching notebook remain byte-identical.

Cache reuse/stale/corrupt/interrupted handling, feature schemas, fold isolation, original fit-label retention, unseen/missing values, ID/target invariance, permutation and twelve malformed CSV cases passed. GPU execution is not covered by this CPU verification. The pinned TabPFN9.2.0 API and official open v2 checkpoint identity were statically checked, but no checkpoint has been downloaded or fitted here.

## Colab handoff

The Lite notebook fixes a browser-editor issue: the old embedded payload occupied one580,616-character line. Compression and line wrapping reduce the file from656,480 to481,311 bytes; all code lines are now at most157 characters. Embedded model/feature/fold sources are unchanged. New transport hashes and fresh setup/audit/feature checks passed. GPU execution remains pending.

A free Tesla T4 connected and pinned packages installed successfully. The automated files.upload transfer failed, and you requested manual upload. Follow COLAB_INSTRUCTIONS.md using V9_Research_Colab_Lite.ipynb and the four CSVs from V9_Official_Inputs.zip. The notebook defaults to trying TabPFN on CUDA, checks full-fold feasibility/reproducibility, and reports failures explicitly. It downloads V9_research_results.zip. Send that ZIP back for review.

Only validation-qualified candidates advance. Any qualified CSV gets a matching standalone coding notebook and separate-process reproduction check before the results archive is delivered. No Olympus upload or one-time coding submission was performed. First place is not established or promised.

## Sources and folder review

The downloaded author folder has the same parsed official data values. Its S08 manifest reports12,088,422.07, worse than V7; the CSV bytes do not match that manifest hash. Its recovery search combines components from different fold assignments; separate confirmation outputs are incomplete. These saved reports were not adopted as our measurements or model predictions.

Model references: [TabPFN documentation](https://docs.priorlabs.ai/models), [official v2 checkpoint](https://huggingface.co/Prior-Labs/TabPFN-v2-reg), [EBM API](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html). TabPFN attribution: Prior Labs; Hollmann et al., Nature2025, doi:10.1038/s41586-024-08328-6.
'''
    (ROOT/'REPORT.md').write_text(report)
    (ROOT/'README.md').write_text('''# V9 manual Colab handoff

Read COLAB_INSTRUCTIONS.md. Upload V9_Research_Colab_Lite.ipynb, select free T4 GPU, Run all, and upload the four official CSVs. Return V9_research_results.zip. No earlier project directories are needed.

CPU EBM screening found no qualified candidate. TabPFN/GPU training is pending manual execution. Preserve scored V7 and its matching coding notebook. No portal upload was made.

Local stages: python improvement_v9/run_local.py STAGE --data-dir /path/to/csvs --out-dir /path/to/results --budget-hours 18
STAGE is audit, pilot, screen, refine, confirm, export or all. After qualified CLI exports, run candidate_notebook.py verify-all in the same pinned environment to generate and freshly execute matching notebooks. The Colab notebook performs that delivery step automatically.

V9 adds interpret-core0.7.8 in its own runtime_deps, preserving V7/V8 dependencies. The official TabPFN package9.2.0 is installed for GPU notebook execution. Model versionv2, repository revision and checkpoint SHA are recorded. Resumable folds reject changed sources, data, folds, training labels, core runtime and GPU execution contracts.

The ZIP omits installed libraries, earlier-source experiment caches, interrupted browser-transfer bundles and large repeated execution directories. The notebook embeds source, verified V7 OOF controls and scored backups. Input CSVs are supplied separately.
''')
    implementation=list(ROOT.glob('*.py'))
    outputs=[ROOT/n for n in ['REPORT.md','README.md','PLAN.md','COLAB_INSTRUCTIONS.md','V9_Research_Colab.ipynb','V9_Research_Colab_Lite.ipynb','V9_Official_Inputs.zip','colab_access.json','portal_trials.csv','reference_manifest.json','v5_contract.json']]
    outputs+=list((ROOT/'recommended_submission').glob('*'))+list((ROOT/'scored_v7_backup').glob('*'))+list((ROOT/'baseline_snapshots').glob('*'))
    outputs+=[p for p in (ROOT/'results').rglob('*') if p.is_file() and 'notebook_execution' not in p.parts and p.name!='independent_delivery_checks.json' and not any(part.startswith('v9_lite_test_') for part in p.parts)]
    manifest={'inputs':e.inputs,'runtime':e.runtime,'implementation':{p.name:sha(p) for p in implementation},
        'outputs':{str(p.relative_to(ROOT)):sha(p) for p in outputs},'decision':json.loads((e.out/'decision.json').read_text()),
        'notebook_execution':execution,'v7_reproduction':json.loads((e.out/'v7_full_reproduction.json').read_text()),
        'preserved_v7':e.manifest['preserved_v7'],'ebm_settings_completed':4,'tabpfn_settings_pending':4,'qualified_candidates':0,'portal_uploads':0}
    save(ROOT/'delivery_manifest.json',manifest)
    with zipfile.ZipFile(ROOT/'V9_Delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in implementation+outputs+[ROOT/'delivery_manifest.json']:z.write(p,p.relative_to(ROOT))
    print('Packaged notebook, official inputs, measured report and manifests.',flush=True)


if __name__=='__main__':main()
