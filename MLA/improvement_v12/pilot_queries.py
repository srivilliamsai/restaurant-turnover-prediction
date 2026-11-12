"""Exact prediction checkpoints for interrupted engineering checks; no fitting."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from core import digest,sha,save,log

def raw_identity(raw):
    # Include every raw column, including IDs/labels, so an isolation check
    # must actually compute its counterfactual instead of returning its control.
    return {'columns':[str(c) for c in raw.columns],'dtypes':[str(t) for t in raw.dtypes],
            'rows':len(raw),'values_and_index_sha256':hashlib.sha256(pd.util.hash_pandas_object(raw,index=True).to_numpy().tobytes()).hexdigest()}

def fitted_identity(fitted):
    return [{'name':a['name'],'builder':a['builder'].state(),'vocabulary':a['vocabulary'],
             'padding_row':raw_identity(a['padding_row'])} for a in fitted['atoms']]

class PilotQueryCache:
    def __init__(self,out,context):
        self.out=Path(out);self.root=self.out/'pilot_query_cache';self.root.mkdir(parents=True,exist_ok=True)
        self.context=context;self.last_inference_seconds=0.
    def predict(self,raw,phase,fit_state,compute):
        assert phase in ['first_fit','fresh_repeat_fit']
        context={'pilot':self.context,'phase':phase,'fitted_preprocessing':fit_state,'raw_query':raw_identity(raw)}
        key=digest(context);p=self.root/(key+'.npz');m=p.with_suffix('.json')
        reused=p.exists() or m.exists()
        if reused:
            assert p.exists() and m.exists(),'Incomplete pilot prediction checkpoint'
            record=json.loads(m.read_text());assert record['context']==context,'Stale pilot prediction checkpoint'
            assert record['sha256']==sha(p),'Corrupt pilot prediction checkpoint'
            with np.load(p,allow_pickle=False) as z:value=z['prediction'].copy()
        else:
            start=time.monotonic();value=np.asarray(compute(),float);seconds=time.monotonic()-start
            assert value.shape==(len(raw),) and np.isfinite(value).all()
            partial=p.with_name(p.stem+'.partial.npz');np.savez_compressed(partial,prediction=value);partial.replace(p)
            record={'context':context,'sha256':sha(p),'inference_seconds':seconds,'rows':len(raw)};save(m,record)
        assert value.shape==(len(raw),) and np.isfinite(value).all()
        self.last_inference_seconds=record['inference_seconds']
        save(self.out/'pilot_query_progress.json',{'setting':self.context['setting'],'phase':phase,
             'completed_query_key':key,'rows':len(raw),'reused':reused,'original_inference_seconds':self.last_inference_seconds,
             'prediction_files':len(list(self.root.glob('*.npz'))),'no_validation_rmse_measured':True})
        log({'PILOT_QUERY_COMPLETE':self.context['setting'],'phase':phase,'rows':len(raw),'reused':reused})
        return value
