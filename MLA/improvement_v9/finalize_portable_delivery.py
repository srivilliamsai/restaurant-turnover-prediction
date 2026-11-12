"""Independently verify and package the recovered V9 delivery; never relax a gate."""
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
import nbformat as nb
import numpy as np
import pandas as pd
import modeling
from engine import Engine, sha, rmse, validate_csv

root=Path(__file__).resolve().parent
run=root/'colab_cli_runs/20261026T160148Z'
source=run/'partial_backup/results'
corrected=run/'portability_confirmation'
draft=run/'recovery_draft'
out=run/'portable_delivery';out.mkdir(exist_ok=True)
progress=out/'delivery_status.json'
def status(value,**more):
    progress.write_text(json.dumps({'status':value,'portal_uploads':0,'paid_resources_requested':False,**more},indent=2)+'\n')
status('waiting_for_required_checks')
started=time.monotonic()
required=[corrected/'decision.json',draft/'fresh_execution_checks.json',draft/'V9_Portable_Colab_executed.ipynb',source/'ebm_backup.json']
while not all(p.exists() for p in required):
    if time.monotonic()-started>3*3600:raise RuntimeError('Recovery allowance exceeded; keep V7.')
    time.sleep(20)
status('independent_verification')
decision=json.loads((corrected/'decision.json').read_text())
assert decision['qualified'], 'Corrected confirmation did not qualify; do not release V9.'
e=Engine(root.parent,out/'verification_runtime')
assert decision['inputs']==e.inputs and decision['sources']==e.sources
assert decision['baseline_platform']['system']=='Darwin'
lock_path=source/'candidate_lock.json'
assert sha(lock_path)==decision['lock_sha256']=='0be8a79c77c85b6b5c29a3efcb238e2d8d9d147e1ae2025e8e218a5246ad9ba5'
lock=json.loads(lock_path.read_text())
assert lock['sources']==e.sources and lock['inputs']==e.inputs
assert lock['plan_sha256']==sha(source/'experiment_plan.json')
pilot=json.loads((source/'pilot.json').read_text())
execution=pilot['families']['tabpfn']['execution_contract']
cache={}
def prediction(name,mode,scheme):
    key=(name,mode,str(scheme))
    if key in cache:return cache[key]
    if name=='v7' and str(scheme) in e.manifest['snapshots']:
        result=e.baseline(scheme)
    else:
        result=np.zeros(len(e.y))
        for number,(a,b) in enumerate(e.splits(scheme)):
            extra={'scheme':str(scheme),'fold':number}
            if name=='v7':
                stem=f'v7_{scheme}_fold{number}'
                directory=corrected/'cache';contract=None
            else:
                stem=f"{name}_{mode}_{str(scheme).replace(':','_')}_fold{number}"
                extra.update(mode=mode,spec=modeling.SPECS[name])
                directory=source/'cache';contract=execution if name.startswith('tabpfn_') else None
            path=directory/(stem+'.npz')
            metadata=json.loads(path.with_suffix('.json').read_text())
            assert sha(path)==metadata['sha256'],stem
            assert metadata['context']==e.context(name,a,b,extra,contract),stem
            values=np.load(path,allow_pickle=False)['prediction']
            assert values.shape==(len(b),) and np.isfinite(values).all() and (values>=0).all()
            result[b]=values
    cache[key]=result
    return result
development=pd.read_csv(source/'development.csv')
assert len(development)==36
diffs=[];segment_rows=[];fold_rows=[]
for row in development.to_dict(orient='records'):
    for scheme in [42,71]:
        b=prediction('v7','plain',scheme)
        c=prediction(row['name'],row['mode'],scheme)
        p=(1-row['weight'])*b+row['weight']*c
        score=rmse(e.y,p);diffs.append(abs(score-row[f'rmse_{scheme}']))
        assert abs(rmse(e.y,b)-score-row[f'gain_{scheme}'])<.01
        common={k:row[k] for k in ['name','mode','weight']}
        for label,mask in [('<=40M',e.y<=40e6),('40-80M',(e.y>40e6)&(e.y<=80e6)),('>80M',e.y>80e6)]:
            segment_rows.append({**common,'scheme':scheme,'segment':label,'rows':int(mask.sum()),'v7_rmse':rmse(e.y[mask],b[mask]),'candidate_rmse':rmse(e.y[mask],p[mask])})
        for number,(_,indices) in enumerate(e.splits(scheme)):
            fold_rows.append({**common,'scheme':scheme,'fold':number,'v7_rmse':rmse(e.y[indices],b[indices]),'candidate_rmse':rmse(e.y[indices],p[indices])})
assert max(diffs)<.01
metrics={};base_errors={};candidate_errors={};concentration=[]
choice=decision['choice']
for scheme in [48211,48212,48213,'strat:27421','city']:
    b=prediction('v7','plain',scheme)
    c=prediction(choice['name'],choice['mode'],scheme)
    p=(1-choice['weight'])*b+choice['weight']*c
    old,new=rmse(e.y,b),rmse(e.y,p)
    recorded=decision['metrics'][str(scheme)]
    assert abs(old-recorded['v7_rmse'])<.01 and abs(new-recorded['candidate_rmse'])<.01
    metrics[str(scheme)]={'v7_rmse':old,'candidate_rmse':new,'gain':old-new,'relative_change':new/old-1}
    reduction=(e.y-b)**2-(e.y-p)**2
    total=float(reduction.sum())
    concentration.append({'scheme':str(scheme),'top10_fraction_of_net_sse_reduction':float(np.sort(reduction)[-10:].sum()/total),
                          'greater80M_fraction_of_net_sse_reduction':float(reduction[e.y>80e6].sum()/total)})
    if isinstance(scheme,int):
        base_errors[scheme]=(e.y-b)**2
        candidate_errors[scheme]=(e.y-p)**2
assert all(metrics[str(s)]['gain']>0 for s in [48211,48212,48213])
assert all(metrics[s]['relative_change']<=.01 for s in ['strat:27421','city'])
rng=np.random.default_rng(8319);gains=[]
for _ in range(5000):
    indices=rng.integers(0,len(e.y),len(e.y))
    gains.append(np.mean([np.sqrt(base_errors[s][indices].mean())-np.sqrt(candidate_errors[s][indices].mean()) for s in [48211,48212,48213]]))
ci=np.percentile(gains,[2.5,97.5]).tolist()
assert np.max(np.abs(np.asarray(ci)-decision['uncertainty']['ci95']))<.01
fresh=json.loads((draft/'fresh_execution_checks.json').read_text())
assert fresh['status']=='PASS' and fresh['maximum_final_difference_INR']<.01
assert fresh['baseline_portable_inference_difference_INR']<.01
assert fresh['baseline_fresh_reproduction_difference_INR']<.01
assert fresh['tabpfn_freshly_fitted_in_this_process']
assert fresh['v7_freshly_retrained_in_this_Colab_process'] is False
assert fresh['inputs']==e.inputs and fresh['sources']==e.sources
assert fresh['execution_contract']==execution and fresh['choice']==choice
assert fresh['baseline_behavior']['malformed_cases_rejected']==12
assert fresh['tabpfn_behavior']['malformed_cases_rejected']==12
csv_path=draft/'fresh_execution.csv'
assert sha(csv_path)==fresh['csv_sha256']
frame=pd.read_csv(csv_path);validate_csv(frame,e.test)
base=pd.read_csv(root/'scored_v7_backup/submission.csv')['Annual Turnover'].to_numpy()
expected=pd.read_csv(draft/'expected.csv')['Annual Turnover'].to_numpy()
assert np.max(np.abs(frame['Annual Turnover'].to_numpy()-expected))<.01
delta=float(np.sqrt(np.mean((frame['Annual Turnover'].to_numpy()-base)**2)))
bound=max(0,12024316.5355221-delta)
book=nb.read(draft/'V9_Portable_Colab_executed.ipynb',as_version=4)
original_book=nb.read(draft/'V9_Portable_Colab.ipynb',as_version=4)
code_cells=[c for c in book.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code_cells)
assert all(not any(o.output_type=='error' for o in c.outputs) for c in code_cells)
assert [c.source for c in code_cells]==[c.source for c in original_book.cells if c.cell_type=='code']
book.cells[0].source=book.cells[0].source.replace('**Draft until the accompanying corrected confirmation report passes.**','**Passed the corrected confirmation and fresh execution checks. Actual portal RMSE is unknown.**')
table=['## Measured confirmation comparisons','',
       'V7 was fitted on its original platform for these comparisons. No candidate setting or blend weight changed. These reused labels provide conditional robustness evidence, not an untouched external test.','',
       '| Assignment | V7 RMSE | V9 RMSE | Gain |','|---|---:|---:|---:|']
for scheme,m in metrics.items():table.append(f"| {scheme} | {m['v7_rmse']:,.2f} | {m['candidate_rmse']:,.2f} | {m['gain']:,.2f} |")
table+=['',f'Conditional 5,000-resample paired-row 95% interval for mean random-assignment RMSE gain: [{ci[0]:,.2f}, {ci[1]:,.2f}] INR. This excludes search, retraining and distribution uncertainty.']
book.cells.append(nb.v4.new_markdown_cell('\n'.join(table)))
submission=out/'submission';submission.mkdir(exist_ok=True)
stem='submission'
final_csv=submission/(stem+'.csv');final_book=submission/(stem+'.ipynb')
shutil.copyfile(csv_path,final_csv);nb.validate(book);nb.write(book,final_book)
pd.DataFrame(segment_rows).to_csv(out/'development_segments.csv',index=False)
pd.DataFrame(fold_rows).to_csv(out/'development_folds.csv',index=False)
shutil.copyfile(source/'development.csv',out/'development_all_36.csv')
for name in ['segments.csv','folds.csv','decision.json']:
    shutil.copyfile(corrected/name,out/('confirmation_'+name))
review={'status':'PASS','all_original_training_targets':3493,'development_configurations':36,
        'development_scores_recomputed':len(diffs),'maximum_development_difference_INR':max(diffs),
        'choice':choice,'metrics':metrics,'uncertainty_ci95':ci,'uncertainty_repetitions':5000,
        'confirmation_concentration':concentration,'fresh_execution':fresh,
        'conditional_portal_rmse_lower_bound':bound,'rms_prediction_change_from_V7':delta,
        'notebook_code_cells':len(code_cells),'notebook_code_unchanged_by_report_append':True,
        'inputs':e.inputs,'sources':e.sources,'new_portal_scores_observed':0}
(out/'independent_review.json').write_text(json.dumps(review,indent=2)+'\n')
best=development.sort_values('mean_rmse').groupby(['name','mode'],sort=False).head(1)
report=['# V9 recovered delivery','',
        '**One validation-qualified V9 CSV is ready for a portal trial. Current observed portal best remains V7: 12,024,316.5355221.**','',
        'The frozen ensemble is 50% your own scored V7 model and 50% raw-target TabPFN v2, V7 features, 16 estimators. All 3,493 original training targets were retained. No paid compute or portal upload was performed.','',
        '## Reproduction resolution','',
        'Fresh V7 fitting on the original Mac reproduced the scored file within INR '+f"{fresh['baseline_fresh_reproduction_difference_INR']:.12f}. "
        'Fresh fitting in Colab had differed by up to INR 427,996.18; matching input and target hashes ruled out changed data. Most component predictions matched exactly, while one V5 component and the nested residual fits differed. The precise low-level numerical cause was not established.','',
        'The delivered notebook embeds the user-owned model from that separately verified fresh Mac fit. Loading its fitted state in Colab preserves V7 predictions within INR '+f"{fresh['baseline_portable_inference_difference_INR']:.12f}. "
        'It does not claim that V7 is retrained identically on Linux. V7 training source and provenance are included; TabPFN trains afresh on every notebook execution. No expected prediction vector is used as a predictor.','',
        'The three random confirmation assignments were recomputed using fresh Mac V7 fits and the unchanged, hash-verified TabPFN predictions. Stratified/city checks use their verified original-platform V7 snapshots. The frozen choice and gates were unchanged.','',
        '## Development','',
        'Nine settings and 36 blend configurations were evaluated. Each row below is the best development blend for one setting; every configuration, including failures, is retained in development_all_36.csv.','',
        '| Setting | Mode | Candidate weight | Gain 42 | Gain 71 |','|---|---|---:|---:|---:|']
for row in best.to_dict(orient='records'):report.append(f"| {row['name']} | {row['mode']} | {row['weight']:.0%} | {row['gain_42']:,.2f} | {row['gain_71']:,.2f} |")
report+=['']+table+['','All three random gains are positive. Stratified and city results satisfy the maximum 1% deterioration gate. No alternative EBM candidate qualified.','',
        '## Error concentration','']
for entry in concentration:report.append(f"- {entry['scheme']}: the ten largest positive row contributions represent {entry['top10_fraction_of_net_sse_reduction']:.1%} of net squared-error reduction; the >80M group contributes {entry['greater80M_fraction_of_net_sse_reduction']:.1%}. Negative values mean deterioration.")
report+=['','All segment results (≤40M, 40–80M, >80M) and fold results accompany this report. Segment diagnostics never determine inference-time rules.','',
        '## Submission and limits','',
        f"CSV: submission/{stem}.csv\n\nMatching executed notebook: submission/{stem}.ipynb",'',
        f"Fresh standalone notebook maximum prediction difference: INR {fresh['maximum_final_difference_INR']:.12f}. "
        'Both components passed row-permutation, ID/target invariance, missing/unseen-input and malformed-CSV checks. The final CSV has 500 unique IDs in original order, exact headers, and finite nonnegative values.','',
        f'RMS prediction distance from scored V7 is INR {delta:,.2f}. Assuming the stated ordinary RMSE on all the same 500 rows, the reverse triangle inequality bounds this candidate’s portal RMSE below by INR '+f'{bound:,.2f}. '
        'Consequently this particular candidate cannot beat the supplied 4,796,740.56 leader score under those assumptions. The bound is not a forecast of its actual score.','',
        'Upload only the CSV for a portal trial, record the returned RMSE against its exact hash, and keep whichever observed file scores lower. Preserve the one-time coding upload until the final choice. Deadline displayed in the supplied portal: October 29, 11:59PM; timezone remains unconfirmed.','',
        'TabPFN uses the official Prior Labs general-purpose v2 regression checkpoint; attribution: Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. No competitor predictions or competition-trained models are used.']
(out/'V9_Report.md').write_text('\n'.join(report)+'\n')
evidence=[out/'V9_Report.md',out/'independent_review.json',out/'development_all_36.csv',
          out/'development_segments.csv',out/'development_folds.csv',out/'confirmation_segments.csv',out/'confirmation_folds.csv',out/'confirmation_decision.json',final_csv,final_book]
manifest={'inputs':e.inputs,'source_hashes':e.sources,'choice':choice,
          'candidate_lock_sha256':decision['lock_sha256'],'v7_checkpoint_sha256':fresh['v7_checkpoint_sha256'],
          'tabpfn_execution_contract':execution,'observed_portal_best':{'version':'V7','rmse':12024316.5355221},
          'portal_uploads':0,'paid_resources_requested':False,
          'files':{str(p.relative_to(out)):sha(p) for p in evidence}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=root/'V9_Portable_Delivery.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in evidence+[out/'manifest.json']:z.write(p,p.relative_to(out))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
status('completed',archive=str(archive),archive_sha256=sha(archive),qualified_candidates=1,
       csv=str(final_csv),notebook=str(final_book),report=str(out/'V9_Report.md'))
(root/'CLI_PROGRESS.md').write_text('# V9 recovery complete\n\nOne qualified V9 trial and its matching executed notebook are ready. The original failed Colab retraining run remains preserved. The recovered delivery uses the independently reproduced V7 fitted checkpoint and freshly trained TabPFN.\n\n'
    f'Report: [V9_Report.md]({out/"V9_Report.md"})\n\n'
    f'CSV: [{stem}.csv]({final_csv})\n\n'
    f'Notebook: [{stem}.ipynb]({final_book})\n\n'
    f'Package: [V9_Portable_Delivery.zip]({archive})\n\n'
    'Actual V9 portal RMSE is unknown. Keep scored V7 until a better score is observed. This candidate cannot beat the supplied first-place score under the stated same-500-row RMSE assumptions.\n')
print('RECOVERED_DELIVERY_COMPLETE',str(out/'V9_Report.md'),flush=True)
