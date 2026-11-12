"""V12 atomic checkpoints, original-data identity and bounded compute ledger."""
import hashlib,json,time,datetime
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import legacy_models,features

ROOT=Path(__file__).resolve().parent
STEM='submission'
DEVELOPMENT=[42,71]
CONFIRM=[71501,71502,71503,'strat:27421','city']
PINNED={'numpy':'2.5.4','pandas':'3.0.6','scikit-learn':'1.9.1','scipy':'1.18.1','catboost':'1.2.10','interpret-core':'0.7.8','tabpfn':'9.2.0','torch':'2.11.0+cu130','pytabkit':'1.7.3'}
MODEL_FILES=['baseline.py','legacy_models.py','legacy_reference.py','frozen_v9_modeling.py','features.py','modeling.py','core.py','verify.py','workflow.py','fresh_worker.py','refit_controls.py','pilot_queries.py']
CAPS={'audit':3600,'pilot':21600,'screen':10800,'confirm':28800,'export':10800}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);temp=p.with_name(p.name+'.partial')
    temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def log(v):print(v if isinstance(v,str) else json.dumps(v,allow_nan=False),flush=True)
def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
class BudgetExceeded(RuntimeError):pass

class Cache:
    def __init__(self,out,hours=18):
        assert 0<hours<=18
        self.out=Path(out);self.out.mkdir(exist_ok=True,parents=True);self.root=self.out/'cache';self.root.mkdir(exist_ok=True)
        self.ledger=self.out/'compute_budget.json';self.hours=hours;self.stage='audit'
    def state(self):return json.loads(self.ledger.read_text()) if self.ledger.exists() else {'seconds':0.,'by_stage':{},'maximum_hours':self.hours}
    @contextmanager
    def charge(self,stage=None):
        stage=stage or self.stage;s=self.state();limit=self.hours*3600-(0 if stage=='export' else 10800)
        combined=s['by_stage'].get('audit',0)+s['by_stage'].get('pilot',0)+s['by_stage'].get('setup',0)
        # User approved eighteen hours and completion of all four settings.
        # Previous charges and the three-hour delivery reserve remain intact.
        if s['seconds']>=limit or s['by_stage'].get(stage,0)>=CAPS.get(stage,64800) or (stage in ['audit','pilot'] and combined>=21600):raise BudgetExceeded(stage)
        cutoff=datetime.datetime.fromisoformat('2026-10-29T21:00:00+05:30' if stage=='export' else '2026-10-29T18:00:00+05:30').timestamp()
        if time.time()>=cutoff:raise BudgetExceeded('Absolute deadline')
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
    def __init__(self,data_dir=None,out_dir=None,hours=18):
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
    def atomic_component(self,name,scheme):
        import modeling
        key=str(scheme)
        registry=json.loads((ROOT/'baseline_registry.json').read_text())
        if name=='BASE' and key in registry['schemes']:
            rec=registry['schemes'][key];assert sha(ROOT/rec['file'])==rec['sha256']
            assert rec['fold_sha256']==digest([b.tolist() for _,b in self.splits(scheme)])
            with np.load(ROOT/rec['file'],allow_pickle=False) as z:return z['t9'].copy()
        p=np.zeros(len(self.y))
        for f,(a,b) in enumerate(self.splits(scheme)):
            context=self.context(name,scheme,f,a,b)
            def compute(a=a,b=b):
                fitted=modeling.fit_atomic(self.train.iloc[a],self.y[a],name,log)
                v=modeling.predict_atomic(fitted,self.train.iloc[b]);extra={'feature_map_sha256':digest(fitted['builder'].state()),'vocabulary_sha256':digest(fitted.get('vocabulary',{}))}
                modeling.release(fitted);return v,extra
            v=self.cache.get(f'{name}_{scheme}_fold{f}',context,compute);assert v.shape==(len(b),);p[b]=v
            log({'atom':name,'scheme':str(scheme),'fold_complete':f+1,'folds':5})
        return p
    def component(self,name,scheme):
        return np.mean([self.atomic_component(atom,scheme) for atom in features.SPECS[name]['atoms']],axis=0)
    def reference_parts(self,scheme):
        key=str(scheme);registry=json.loads((ROOT/'baseline_registry.json').read_text())
        if key in registry['schemes']:
            rec=registry['schemes'][key];assert sha(ROOT/rec['file'])==rec['sha256']
            assert rec['fold_sha256']==digest([v.tolist() for _,v in self.splits(scheme)]) and rec['inputs']==self.inputs and rec['row_ids_sha256']==self.row_hash
            with np.load(ROOT/rec['file'],allow_pickle=False) as z:return z['v7'].copy(),z['t9'].copy()
        base=self.reference(scheme);t=self.atomic_component('BASE',scheme);return 2*base-t,t
    def reference(self,scheme):
        registry=json.loads((ROOT/'baseline_registry.json').read_text());key=str(scheme)
        if key in registry['schemes']:
            rec=registry['schemes'][key];p=ROOT/rec['file'];assert sha(p)==rec['sha256'];assert rec['fold_sha256']==digest([v.tolist() for _,v in self.splits(scheme)])
            assert rec['row_ids_sha256']==self.row_hash and rec['inputs']==self.inputs
            with np.load(p,allow_pickle=False) as z:return z['v9'].copy()
        t=self.atomic_component('BASE',scheme)
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
