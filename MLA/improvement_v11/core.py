"""V11 atomic checkpoints, original-data identity and bounded compute ledger."""
import hashlib,json,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import legacy_models,features

ROOT=Path(__file__).resolve().parent
STEM='submission'
DEVELOPMENT=[42,71]
CONFIRM=[61301,61302,61303,'strat:27421','city']
PINNED={'numpy':'2.5.4','pandas':'3.0.6','scikit-learn':'1.9.1','scipy':'1.18.1','catboost':'1.2.10','interpret-core':'0.7.8','tabpfn':'9.2.0','torch':'2.11.0+cu130','tabicl':'2.2.0'}
MODEL_FILES=['baseline.py','legacy_models.py','legacy_reference.py','frozen_v9_modeling.py','features.py','modeling.py','core.py','verify.py','workflow.py','fresh_worker.py','refit_controls.py']
CAPS={'audit':3600,'pilot':3600,'screen':7200,'confirm':10800,'export':7200}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);temp=p.with_name(p.name+'.partial')
    temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def log(v):print(v if isinstance(v,str) else json.dumps(v,allow_nan=False),flush=True)
def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
class BudgetExceeded(RuntimeError):pass

class Cache:
    def __init__(self,out,hours=8):
        assert 0<hours<=8
        self.out=Path(out);self.out.mkdir(exist_ok=True,parents=True);self.root=self.out/'cache';self.root.mkdir(exist_ok=True)
        self.ledger=self.out/'compute_budget.json';self.hours=hours;self.stage='audit'
    def state(self):return json.loads(self.ledger.read_text()) if self.ledger.exists() else {'seconds':0.,'by_stage':{},'maximum_hours':self.hours}
    @contextmanager
    def charge(self,stage=None):
        stage=stage or self.stage;s=self.state();limit=self.hours*3600-(0 if stage=='export' else 7200)
        combined=s['by_stage'].get('audit',0)+s['by_stage'].get('pilot',0)+s['by_stage'].get('setup',0)
        if s['seconds']>=limit or s['by_stage'].get(stage,0)>=CAPS.get(stage,28800) or (stage in ['audit','pilot'] and combined>=3600):raise BudgetExceeded(stage)
        start=time.monotonic()
        try:yield
        finally:
            s=self.state();elapsed=time.monotonic()-start;s['seconds']+=elapsed;s['by_stage'][stage]=s['by_stage'].get(stage,0)+elapsed
            s.update(maximum_hours=self.hours,measurement='Task computation/setup wall seconds; retained across resumes; local Mac receipts added once.');save(self.ledger,s)
    def get(self,key,context,compute):
        p=self.root/(key.replace(':','_')+'.npz');m=p.with_suffix('.json')
        if p.exists() or m.exists():
            if not(p.exists() and m.exists()):raise ValueError('Incomplete checkpoint '+key)
            record=json.loads(m.read_text());assert record['context']==context,'Stale checkpoint '+key;assert record['sha256']==sha(p),'Corrupt checkpoint '+key
            with np.load(p,allow_pickle=False) as z:v=z['prediction'].copy()
        else:
            with self.charge():result=compute()
            v,extra=result if isinstance(result,tuple) else (result,{})
            v=np.asarray(v,float);assert v.ndim==1 and np.isfinite(v).all()
            temp=p.with_name(p.stem+'.partial.npz');np.savez_compressed(temp,prediction=v);temp.replace(p)
            save(m,{'context':context,'sha256':sha(p),**extra})
        assert v.ndim==1 and np.isfinite(v).all();return v

class Engine:
    def __init__(self,data_dir=None,out_dir=None,hours=8):
        self.data=Path(data_dir or ROOT.parent).resolve();self.out=Path(out_dir or ROOT/'results').resolve()
        self.train,self.test,self.y=legacy_models.load_data(self.data)
        self.manifest=json.loads((ROOT/'reference_manifest.json').read_text());self.inputs={n:sha(self.data/n) for n in legacy_models.REQUIRED}
        assert self.inputs==self.manifest['inputs'],'Official inputs changed'
        self.sources={n:sha(ROOT/n) for n in MODEL_FILES};self.sources.update({n:sha(ROOT/n) for n in ['city_map.json','v5_contract.json','reference_manifest.json','baseline_registry.json']})
        self.cache=Cache(self.out,hours);self.row_hash=digest(self.train['Registration Number'].tolist())
    def splits(self,scheme):
        if str(scheme)!='city':return legacy_models.split_indices(self.train,scheme)
        groups=features.canonical_city(self.train['City']).fillna('__V10_UNKNOWN_CITY__')
        result=list(GroupKFold(5).split(self.train,groups=groups))
        assert all(set(groups.iloc[a]).isdisjoint(groups.iloc[b]) for a,b in result)
        return result
    def context(self,name,scheme,fold,a,b):
        config=json.loads((ROOT/'resolved_models.json').read_text())
        return {'name':name,'scheme':str(scheme),'fold':fold,'inputs':self.inputs,'sources':self.sources,'model_contract':config,
                'training_rows':digest(a.tolist()),'validation_rows':digest(b.tolist()),'target_hash':hashlib.sha256(self.y[a].tobytes()).hexdigest(),
                'row_order':self.row_hash,'fold_hash':digest([v.tolist() for _,v in self.splits(scheme)])}
    def component(self,name,scheme):
        import modeling
        p=np.zeros(len(self.y))
        for f,(a,b) in enumerate(self.splits(scheme)):
            context=self.context(name,scheme,f,a,b)
            def compute(a=a,b=b):
                fitted=modeling.fit(self.train.iloc[a],self.y[a],name,log);v=modeling.predict(fitted,self.train.iloc[b])
                extra={'feature_map_sha256':digest(fitted['builder'].state()),'vocabulary_sha256':digest(fitted['vocabulary'])}
                modeling.release(fitted)
                return v,extra
            v=self.cache.get(f'{name}_{scheme}_fold{f}',context,compute);assert v.shape==(len(b),);p[b]=v
            log({'setting':name,'scheme':str(scheme),'fold_complete':f+1,'folds':5})
        return p
    def reference(self,scheme):
        registry=json.loads((ROOT/'baseline_registry.json').read_text());key=str(scheme)
        if key in registry['schemes']:
            rec=registry['schemes'][key];p=ROOT/rec['file'];assert sha(p)==rec['sha256'];assert rec['fold_sha256']==digest([v.tolist() for _,v in self.splits(scheme)])
            assert rec['row_ids_sha256']==self.row_hash and rec['inputs']==self.inputs
            with np.load(p,allow_pickle=False) as z:return z['v9'].copy()
        t=self.component('BASE',scheme)
        path=self.out/f'reference_v7_{scheme}.npz';meta=path.with_suffix('.json')
        if not(path.exists() and meta.exists()):raise FileNotFoundError('Original Mac control pending '+key)
        rec=json.loads(meta.read_text());assert rec['sha256']==sha(path) and rec['inputs']==self.inputs
        assert rec['row_ids_sha256']==self.row_hash and rec['fold_sha256']==digest([v.tolist() for _,v in self.splits(scheme)])
        assert rec['sources']=={n:sha(ROOT/n) for n in ['baseline.py','legacy_models.py','legacy_reference.py','v5_contract.json']}
        assert rec['platform']['system']=='Darwin' and rec['lock_sha256']==sha(self.out/'candidate_lock.json')
        with np.load(path,allow_pickle=False) as z:b=z['prediction'].copy()
        return .5*b+.5*t

def validate_csv(frame,test):
    assert frame.shape==(500,2) and frame.columns.tolist()==['Registration Number','Annual Turnover']
    assert frame['Registration Number'].equals(test['Registration Number'].reset_index(drop=True)) and frame['Registration Number'].is_unique
    p=frame['Annual Turnover'].to_numpy(float);assert np.isfinite(p).all() and (p>=0).all()
