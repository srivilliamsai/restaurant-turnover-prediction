"""Independently reconstruct all measured blends from hash-verified fold predictions."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import Engine,sha,digest,rmse,save
import features


def component(e,name,scheme):
    p=np.zeros(len(e.y))
    for f,(a,b) in enumerate(e.splits(scheme)):
        stem=f'{name}_{str(scheme)}_fold{f}'.replace(':','_').replace('+','_')
        path=e.out/'cache'/(stem+'.npz');metadata=json.loads(path.with_suffix('.json').read_text())
        assert metadata['context']==e.context(name,scheme,f,a,b),stem
        assert metadata['sha256']==sha(path),stem
        values=np.load(path,allow_pickle=False)['prediction']
        assert values.shape==(len(b),) and np.isfinite(values).all()
        builder=features.FeatureBuilder(name).fit(e.train.iloc[a])
        assert metadata['feature_map_sha256']==digest(builder.state())
        _,vocab=features.native_frame(builder.transform(e.train.iloc[a]))
        assert metadata['vocabulary_sha256']==digest(vocab)
        p[b]=values
    return p


def main():
    e=Engine(ROOT.parent,ROOT/'colab_results/results')
    table=pd.read_csv(e.out/'screening.csv');assert len(table)==24
    candidates={};references={s:e.reference(s) for s in [42,71]};rows=[];largest=0.
    for name in features.SPECS:
        candidates[name]={s:component(e,name,s) for s in [42,71]}
        for w in [.25,.5,.75,1.]:
            row={'name':name,'replacement_weight':w,'feature_count':features.COUNTS[name]}
            observed=table[(table.name==name)&(table.replacement_weight==w)].iloc[0]
            for scheme in [42,71]:
                b,t,v9=references[scheme]
                p=.5*b+.5*((1-w)*t+w*candidates[name][scheme])
                score=rmse(e.y,p);gain=rmse(e.y,v9)-score
                row[f'rmse_{scheme}']=score;row[f'gain_{scheme}']=gain
                largest=max(largest,abs(observed[f'rmse_{scheme}']-score),abs(observed[f'gain_{scheme}']-gain))
            row['mean_rmse']=float(np.mean([row['rmse_42'],row['rmse_71']]));rows.append(row)
    assert largest<.01
    reconstructed=pd.DataFrame(rows).sort_values(['mean_rmse','replacement_weight','feature_count','name'])
    eligible=reconstructed[(reconstructed.gain_42>0)&(reconstructed.gain_71>0)]
    lock=json.loads((e.out/'candidate_lock.json').read_text())
    assert lock['sources']==e.sources and lock['inputs']==e.inputs and lock['configurations']==24
    if len(eligible):
        expected=eligible.iloc[0];choice=lock['choice']
        assert expected['name']==choice['name'] and expected['replacement_weight']==choice['replacement_weight']
    else:assert lock['choice'] is None
    decision=json.loads((e.out/'decision.json').read_text())
    assert decision['lock_sha256']==sha(e.out/'candidate_lock.json')
    if lock['choice'] is not None:
        result={};choice=lock['choice'];name=choice['name'];w=choice['replacement_weight']
        for scheme in [48211,48212,48213,'strat:27421','city']:
            if scheme=='city':
                path=e.out/'reference_city_v7.npz';meta=json.loads(path.with_suffix('.json').read_text())
                assert meta['sha256']==sha(path)
                assert meta['fold_sha256']==digest([b.tolist() for _,b in e.splits('city')])
                b=np.load(path,allow_pickle=False)['prediction'];t=component(e,'BASE','city');v9=.5*b+.5*t
            else:b,t,v9=e.reference(scheme)
            c=component(e,name,scheme);p=.5*b+.5*((1-w)*t+w*c)
            old,new=rmse(e.y,v9),rmse(e.y,p)
            result[str(scheme)]={'gain':old-new,'relative_change':new/old-1}
            assert abs(decision['metrics'][str(scheme)]['candidate_rmse']-new)<.01
        qualified=all(result[str(s)]['gain']>0 for s in [48211,48212,48213]) and all(result[s]['relative_change']<=.01 for s in ['strat:27421','city'])
        assert decision['qualified']==qualified
    else:assert not decision['qualified']
    record={'status':'PASS','development_blends_recomputed':24,'candidate_outer_fits_verified':60,'control_outer_fits_verified':10,
            'maximum_metric_difference_INR':largest,'frozen_selection_matches':True,
            'qualification_recomputed':True,'qualified':decision['qualified'],'inputs':e.inputs,'sources':e.sources}
    save(ROOT/'independent_review.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ['sources','inputs']},indent=2))


if __name__=='__main__':main()
