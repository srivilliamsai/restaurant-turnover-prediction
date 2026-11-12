"""Independent V9 score arithmetic, provenance and standalone notebook checks."""
import json
import zipfile
import numpy as np
import pandas as pd
import nbformat as nb
from engine import Engine, ROOT, sha, save, rmse


def main():
    e=Engine();m=json.loads((ROOT/'delivery_manifest.json').read_text())
    for category in ['implementation','outputs']:
        for name,value in m[category].items():assert sha(ROOT/name)==value,name
    for name,value in m['inputs'].items():assert sha(ROOT.parent/name)==value,name
    for name,value in e.manifest['preserved_v7'].items():
        for directory in [ROOT/'scored_v7_backup',ROOT/'recommended_submission',ROOT.parent/'improvement_v7/candidates/trial_1']:
            assert sha(directory/name)==value
    table=pd.read_csv(e.out/'screening.csv');assert len(table)==16 and table.name.nunique()==4
    assert not ((table.gain_42>0)&(table.gain_71>0)).any()
    count=0
    for r in table.to_dict('records'):
        for s in [42,71]:
            b=e.baseline(s);p=np.zeros(len(e.y))
            for f,(a,v) in enumerate(e.splits(s)):
                path=e.out/'cache'/f'{r["name"]}_plain_{s}_fold{f}.npz'
                meta=json.loads(path.with_suffix('.json').read_text())
                assert sha(path)==meta['sha256']
                assert meta['context']==e.context(r['name'],a,v,{'scheme':str(s),'fold':f,'mode':'plain','spec':__import__('modeling').SPECS[r['name']]})
                p[v]=np.load(path,allow_pickle=False)['prediction']
            score=rmse(e.y,(1-r['weight'])*b+r['weight']*p)
            assert abs(score-r[f'rmse_{s}'])<.01
            assert abs(rmse(e.y,b)-score-r[f'gain_{s}'])<.01
            count+=1
    segments=pd.read_csv(e.out/'development_segments.csv');assert len(segments)==96
    for r in table.to_dict('records'):
        for s in [42,71]:
            q=segments[(segments.name==r['name'])&(segments.weight==r['weight'])&(segments.scheme==s)]
            assert q.rows.sum()==len(e.y)
            assert abs(np.sqrt(q.candidate_sse.sum()/len(e.y))-r[f'rmse_{s}'])<.01
    lock=json.loads((e.out/'candidate_lock.json').read_text());decision=json.loads((e.out/'decision.json').read_text())
    assert lock['shortlist']==[] and decision['trials']==[] and not (e.out/'confirmation.csv').exists()
    book=nb.read(ROOT/'V9_Research_Colab.ipynb',as_version=4);nb.validate(book)
    cells=[c for c in book.cells if c.cell_type=='code']
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in cells)
    assert 'RUN_TABPFN = os.environ.get' in cells[0].source and "PACKAGES['tabpfn']" in cells[0].source
    assert m['notebook_execution']['gpu_branch_executed'] is False
    assert m['notebook_execution']['prediction_checkpoint_reads']==0
    assert m['v7_reproduction']['maximum_difference_INR']<.01
    with zipfile.ZipFile(ROOT/'V9_Delivery.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for name,value in m['outputs'].items():
            import hashlib
            assert hashlib.sha256(z.read(name)).hexdigest()==value,name
    record={'status':'PASS','independent_development_scores':count,'all_original_labels':True,
            'segment_rows':96,'fold_rows':160,'fresh_cpu_notebook_executed':True,
            'gpu_execution_pending_user':True,'qualified_csvs':0,'v7_preserved':True,'portal_uploads':0}
    save(e.out/'independent_delivery_checks.json',record)
    with zipfile.ZipFile(ROOT/'V9_Delivery.zip','a',zipfile.ZIP_DEFLATED) as z:
        z.write(e.out/'independent_delivery_checks.json','results/independent_delivery_checks.json')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
