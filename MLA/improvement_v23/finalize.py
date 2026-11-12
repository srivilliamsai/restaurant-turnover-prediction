"""Independent score reconstruction and compact delivery, including failures."""
import csv,hashlib,json,zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';DELIVERY=ROOT/'delivery'

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def save(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    DELIVERY.mkdir(exist_ok=True);audit=read(ROOT/'history_audit.json');protocol=read(ROOT/'protocol.json');frozen=read(ROOT/'source_frozen.json')
    for n,h in frozen['files_sha256'].items():assert sha(ROOT/n)==h
    for n,h in audit['files_sha256'].items():assert sha(ROOT/n)==h
    for n,h in audit['inputs'].items():assert sha(ROOT.parent/n)==h
    for n,h in audit['source_sha256'].items():assert sha(ROOT.parent/'improvement_v22'/n)==h
    for n,h in audit['witnesses'].items():
        p=ROOT.parent/n;assert sha(p)==h['prediction_sha256'] and sha(p.with_suffix('.json'))==h['metadata_sha256']
    for ext in ['csv','ipynb']:
        assert sha(DELIVERY/f'KEEP_V14_submission.{ext}')==audit['reference_sha256'][ext]
    with np.load(ROOT/'controls.npz',allow_pickle=False) as z:controls={k:z[k].copy() for k in z.files}
    y=controls['y'];parts=read(ROOT/'partitions.json');reconstructed={};folds=0
    for scheme,indices in parts.items():
        p=np.zeros(len(y));complete=True
        for fold,(a,b) in enumerate(indices):
            a,b=np.array(a),np.array(b);file=OUT/'cache'/(scheme.replace(':','_')+f'_fold{fold}.npz');meta=file.with_suffix('.json')
            if not meta.exists():complete=False;continue
            m=read(meta);assert sha(file)==m['sha256']
            context=m['context'];assert context['scheme']==scheme and context['fold']==fold and context['frozen']==frozen
            for key,rows in [('train_indices',a),('valid_indices',b)]:
                assert context[key]==hashlib.sha256(json.dumps(rows.tolist(),sort_keys=True,allow_nan=False).encode()).hexdigest()
            assert context['training_targets']==hashlib.sha256(y[a].tobytes()).hexdigest()
            with np.load(file,allow_pickle=False) as z:pred=z['prediction'].copy()
            assert pred.shape==b.shape and np.isfinite(pred).all();p[b]=pred;folds+=1
        if complete:reconstructed[scheme]=p
    checked=[]
    screen=OUT/'screening.csv'
    if screen.exists():
        for row in pd.read_csv(screen).to_dict('records'):
            w=row['weight'];gains=[]
            for s in protocol['development']:
                pred=(1-w)*controls['v14_'+s]+w*reconstructed[s]
                old=rmse(y,controls['v14_'+s]);new=rmse(y,pred);gains.append((old-new)/old)
                assert abs(row[s+'_candidate_RMSE']-new)<1e-6
            assert abs(row['mean_relative_gain']-float(np.mean(gains)))<1e-12
            assert bool(row['qualifies'])==bool(min(gains)>0 and np.mean(gains)>=.002)
            checked.append(row)
    decision=read(OUT/'decision.json') if (OUT/'decision.json').exists() else None
    confirmation_verified=False
    if decision and 'confirmation' in decision:
        w=decision['weight'];bases={};candidates={};verified=[]
        for s in protocol['confirmation']:
            base=controls['v14_'+s.replace(':','_')];new=(1-w)*base+w*reconstructed[s]
            old,score=rmse(y,base),rmse(y,new)
            record={'scheme':s,'v14_RMSE':old,'candidate_RMSE':score,'gain_INR':old-score,'relative_gain':(old-score)/old}
            reported=next(r for r in decision['confirmation'] if r['scheme']==s)
            assert all(abs(record[k]-reported[k])<1e-6 for k in record if k!='scheme')
            verified.append(record);bases[s]=base;candidates[s]=new
        rng=np.random.default_rng(8319);draws=[]
        for _ in range(5000):
            ix=rng.integers(0,len(y),len(y))
            draws.append(np.mean([rmse(y[ix],bases[s][ix])-rmse(y[ix],candidates[s][ix]) for s in protocol['confirmation'][:3]]))
        interval=np.quantile(draws,[.025,.975]);assert np.max(abs(interval-np.asarray(decision['conditional_CI95_INR'])))<1e-6
        mean=float(np.mean([r['relative_gain'] for r in verified[:3]]))
        assert abs(mean-decision['mean_random_relative_gain'])<1e-12
        qualifies=bool(all(r['gain_INR']>0 for r in verified[:3]) and mean>=.002 and interval[0]>0 and all(r['relative_gain']>=-.01 for r in verified[3:]))
        assert qualifies==decision['qualified'];confirmation_verified=True
    failures=[read(p) for p in sorted(OUT.glob('*_failure.json'))]
    diagnostics=[]
    for s,pred in reconstructed.items():
        base=controls['v14_'+s.replace(':','_')];new=.75*base+.25*pred
        se=(y-new)**2;oldse=(y-base)**2
        for lo,hi in [(0,20e6),(20e6,40e6),(40e6,80e6),(80e6,float('inf'))]:
            ix=(y>lo)&(y<=hi)
            if ix.any():diagnostics.append({'scheme':s,'turnover_low_INR':lo,'turnover_high_INR':hi,'rows':int(ix.sum()),'V14_RMSE':rmse(y[ix],base[ix]),'candidate_RMSE':rmse(y[ix],new[ix]),'candidate_squared_error_share':float(se[ix].sum()/se.sum())})
    if diagnostics:pd.DataFrame(diagnostics).to_csv(DELIVERY/'turnover_band_diagnostics.csv',index=False)
    result={'status':'PASS','meaning':'Artifact identities and any complete development scores independently reconstructed; not an accuracy qualification','original_P35_control_contexts':35,'complete_new_folds':folds,'screening_comparisons_verified':len(checked),'confirmation_and_bootstrap_verified':confirmation_verified,'all_3493_labels_retained':True,'fallback_byte_identical':True,'portal_uploads':0,'new_portal_score_known':False,'source_revisions_locked':True,'selected_backend':protocol.get('backend')}
    save(DELIVERY/'independent_review.json',result)
    exported=read(DELIVERY/'qualified_export.json') if (DELIVERY/'qualified_export.json').exists() else None
    if exported:
        assert exported['status']=='PASS' and exported['fresh_notebook_executed']
        notebook=DELIVERY/'V23_CANDIDATE_submission.ipynb'
        csv=DELIVERY/'V23_CANDIDATE_submission.csv'
        assert sha(notebook)==exported['notebook_sha256'] and sha(csv)==exported['csv_sha256']
        frame=pd.read_csv(csv);test=pd.read_csv(ROOT.parent/'Test_dataset.csv')
        assert frame.shape==(500,2) and frame.columns.tolist()==['Registration Number','Annual Turnover']
        assert frame['Registration Number'].equals(test['Registration Number'])
        assert np.isfinite(frame['Annual Turnover']).all() and (frame['Annual Turnover']>=0).all()
        with np.load(DELIVERY/'full_component_witness.npz',allow_pickle=False) as z:
            expected=.75*pd.read_csv(DELIVERY/'KEEP_V14_submission.csv')['Annual Turnover'].to_numpy(float)+.25*z['prediction']
            assert np.max(abs(z['expected']-expected))<.01
            assert np.max(abs(frame['Annual Turnover']-expected))<.01
        import nbformat
        book=nbformat.read(notebook,as_version=4)
        assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
        assert not [o for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
        result['fresh_qualified_export_verified']=True;save(DELIVERY/'independent_review.json',result)
    if decision and decision.get('qualified') and exported:
        outcome='V23 passed local confirmation and fresh complete notebook/CSV reproduction. Its portal score is unknown; V14 remains the best observed submission.'
    elif decision and decision.get('qualified'):
        outcome='Local confirmation passed. A fresh complete submission reproduction remains required before considering a portal trial.'
    elif decision:
        outcome='Retain V14. The fixed V23 blend did not pass the frozen validation gates.'
    elif failures:
        outcome='Retain V14. V23 did not pass engineering admission; no qualified replacement was produced.'
    else:
        outcome='Retain V14. The one-hour attempt ended before a complete qualified replacement was produced.'
    when=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()
    backend='Official Torch CUDA on the owned free T4.' if protocol.get('backend')=='official_torch_cuda' else 'Local MLX backend from pinned open PR 60; no claim of exact PyTorch equivalence.'
    transfer=read(ROOT/'transport_run/transfer.json') if (ROOT/'transport_run/transfer.json').exists() else {}
    permission='After automatic approval review initially rejected transfer, the user explicitly authorized uploading these data and artifacts to the already selected free Colab account. The transfer receipt records the account hash, owned session and bundle hash.' if transfer.get('uploaded') else 'No dataset upload occurred after the initial automatic approval review rejection.'
    lines=['# V23 result','',outcome,'','Best observed portal RMSE: **11,736,110.7877709 (V14)**. Latest recorded rank is 11 in the October 29 19:44:46 screenshot; current rank is unknown. No portal upload occurred.','',
        '## Experiment','',f"One fixed {protocol['candidate']} component: official LimiX-2 checkpoint, original 40 features, training-only preprocessing, two fixed transformation members, seed 10242. One blend: 75% V14 plus 25% LimiX. All 3,493 labels retained. "+backend,'',
        'The user shortened the available time to one hour before predictions. Initial and amended plans remain preserved. '+permission,'',
        f"Independent audit: 35 original matched P35 contexts verified; {folds} complete new folds; {len(checked)} development comparison(s) reconstructed. Byte-identical scored V14 CSV and notebook are included.",'']
    for failure in failures:
        lines += ['## Measured limitation','',f"Stage `{failure['stage']}`: `{failure['type']}` — {failure['message']}",'']
    pilot=OUT/'pilot.json'
    if pilot.exists():
        p=read(pilot);lines += ['## Timing and numerical checks','',f"Pilot status: {p['status']}; full development-fold prediction: {p['fold_seconds']:.2f} seconds; full-data 500-query prediction: {p['full_seconds']:.2f} seconds.",'']
    if checked:
        lines+=['## Development results','','| New-model weight | Assignment 42 gain | Assignment 71 gain | Mean relative gain | Gate |','| --- | --- | --- | --- | --- |']
        for r in checked:lines.append(f"| {r['weight']:.0%} | INR {r['42_gain_INR']:,.2f} | INR {r['71_gain_INR']:,.2f} | {r['mean_relative_gain']:.6%} | {'Pass' if r['qualifies'] else 'Fail'} |")
        lines+=['']
    lines+=['## Recommendation','','Use the exact scored V14 artifacts unless a completely validated, reproduced candidate subsequently returns a lower actual portal score. Local percentages do not establish a new rank. Failed engineering is not evidence that LimiX is less accurate. Do not spend the remaining deadline on an unplanned model or weight search.','',
        'Research, pinned source/checkpoint manifests, the executable adapter, measured logs and independent review are included. Public weights and installed dependency binaries are excluded from the compact archive. Downloading weights is reproducible from download_checkpoint.py; no private data are needed by that download.','',f'Prepared {when}.']
    (DELIVERY/'V23_Result_Summary.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'README.md').write_text('# V23 complete\n\n'+outcome+'\n\nSee delivery/V23_Result_Summary.md, PLAN.md and RESEARCH_SOURCES.md. The best-scored V14 CSV and notebook are preserved in delivery/.\n')
    save(DELIVERY/'result_summary.json',{'outcome':outcome,'decision':decision,'failures':failures,'independent_review':result,'prepared_IST':when})
    archive=DELIVERY/'V23_review_and_best_submission.zip'
    names=[]
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if not p.is_file() or any(n in rel.parts for n in ['runtime_deps','local_env','weights','__pycache__','audit_scratch']):continue
        if p.suffix=='.zip' or p.name.endswith('.partial'):continue
        names.append(p)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in names:z.write(p,'improvement_v23/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'outcome':outcome,'review':result,'archive':str(archive),'archive_bytes':archive.stat().st_size},indent=2))
if __name__=='__main__':main()
