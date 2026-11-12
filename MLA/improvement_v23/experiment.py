"""Finite, frozen V23 pipeline. All scores use complete original labels."""
import argparse,hashlib,importlib.metadata,json,os,platform,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def log(value):print(json.dumps(value,allow_nan=False),flush=True)
def require(seconds=1,engineering=False):
    a=read(ROOT/'authorization.json');end=a['engineering_stop_epoch'] if engineering else a['training_stop_epoch']
    assert time.time()+seconds<end,'V23 elapsed allowance cannot fit mandatory work'
def status(stage,**kw):save(ROOT/'status.json',{'stage':stage,'epoch':time.time(),**kw})

def identities():
    f=read(ROOT/'source_frozen.json')
    for n,h in f['files_sha256'].items():assert sha(ROOT/n)==h,'Changed frozen input: '+n
    manifest=read(ROOT/'research_sources/source_manifest.json')
    for n,h in manifest['files_sha256'].items():assert sha(ROOT/'research_sources/LimiX'/n)==h,'Changed author source: '+n
    if read(ROOT/'protocol.json').get('backend')=='local_mlx':
        for n,h in read(ROOT/'research_sources/mlx_manifest.json')['files_sha256'].items():
            assert sha(ROOT/'research_sources/mlx_port'/n)==h,'Changed MLX source: '+n
    return f

def dataset():
    audit=read(ROOT/'history_audit.json')
    for n,h in audit['files_sha256'].items():assert sha(ROOT/n)==h
    train=pd.read_csv(ROOT/'feature_train.csv');test=pd.read_csv(ROOT/'feature_test.csv')
    with np.load(ROOT/'controls.npz',allow_pickle=False) as z:controls={k:z[k].copy() for k in z.files}
    assert train.shape==(3493,40) and test.shape==(500,40)
    return train,test,controls,read(ROOT/'partitions.json')

def checkpoint():
    record=read(ROOT/'research_sources/checkpoint_manifest.json');filename='LimiX-2.ckpt'
    want=next(x['lfs']['sha256'] for x in record['files'] if x['rfilename']==filename)
    path=ROOT/'weights'/filename
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path=Path(hf_hub_download(repo_id=record['repo_id'],filename=filename,revision=record['revision'],local_dir=ROOT/'weights',token=False))
        hf_hub_download(repo_id=record['repo_id'],filename='LICENSE',revision=record['revision'],local_dir=ROOT/'weights',token=False)
    assert sha(path)==want
    return path

def resolve():
    require(engineering=True);identities()
    import torch
    local=read(ROOT/'protocol.json').get('backend')=='local_mlx'
    if local:
        import mlx.core as mx
        assert mx.metal.is_available(),'Local Metal backend unavailable'
        gpu='Apple Metal via MLX'
    else:
        assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4'
        gpu=torch.cuda.get_device_name(0)
    path=checkpoint()
    keys=['torch','numpy','pandas','scikit-learn','scipy','einops','kditransform','huggingface-hub','typing_extensions','joblib']+(['mlx'] if local else ['nvtx'])
    save(ROOT/'resolved.json',{'versions':{k:importlib.metadata.version(k) for k in keys},'python':sys.version,'platform':platform.platform(),'gpu':gpu,'cuda':torch.version.cuda,'checkpoint_sha256':sha(path),'source_revision':read(ROOT/'protocol.json')['source_revision'],'mixed_precision':False,'seed':10242,'query_batch_size':read(ROOT/'protocol.json')['query_batch_size']})
    status('resolved');log({'resolve':'PASS','checkpoint_sha256':sha(path)})

def create():
    import torch
    import adapter
    r=read(ROOT/'resolved.json')
    assert all(importlib.metadata.version(k)==v for k,v in r['versions'].items())
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    return adapter.make_predictor(checkpoint())

def pilot():
    require(engineering=True);identities()
    import torch,adapter
    train,test,c,partitions=dataset();a,b=map(np.asarray,partitions['42'][0]);y=c['y']
    started=time.time();model=create();state=adapter.fit_state(train.iloc[a],y[a]);query=train.iloc[b]
    prep=adapter.preprocess_audit(model,state,query)
    save(RESULTS/'pilot_progress.json',{'step':'preprocessing_passed','details':prep,'epoch':time.time()})
    t=time.time();p=adapter.infer(model,state,query);fold_seconds=time.time()-t
    save(RESULTS/'pilot_progress.json',{'step':'first_fold_prediction','fold_seconds':fold_seconds,'epoch':time.time()})
    adapter.save_state(state,RESULTS/'serialized_state.npz');restored=adapter.load_state(RESULTS/'serialized_state.npz')
    require(1.3*fold_seconds*8,engineering=True)
    repeated=adapter.infer(model,restored,query)
    same=float(np.max(abs(p-repeated)))
    assert same<.01,'Serialized-state reproduction exceeded INR 0.01'
    order=np.random.default_rng(2301).permutation(len(query))
    perm=adapter.infer(model,state,query.iloc[order])[np.argsort(order)]
    split=np.r_[adapter.infer(model,state,query.iloc[:len(query)//2]),adapter.infer(model,state,query.iloc[len(query)//2:])]
    single=np.array([adapter.infer(model,state,query.iloc[[i]])[0] for i in range(12)])
    diffs={'permutation':float(np.max(abs(p-perm))),'split':float(np.max(abs(p-split))),'individual_12':float(np.max(abs(p[:12]-single)))}
    drifts={k:abs(rmse(y[b],p)-rmse(y[b],v)) for k,v in [('permutation',perm),('split',split)]}
    drifts['individual_12']=abs(rmse(y[b[:12]],p[:12])-rmse(y[b[:12]],single))
    save(RESULTS/'pilot_progress.json',{'step':'query_checks_measured','same_state_difference_INR':same,'differences_INR':diffs,'RMSE_drifts_INR':drifts,'epoch':time.time()})
    assert max(diffs.values())<=10 and max(drifts.values())<=1,'Query numerical behavior failed fixed tolerance'
    adapter.release(model);del model
    fresh=create();freshp=adapter.infer(fresh,restored,query);fresh_diff=float(np.max(abs(p-freshp)))
    assert fresh_diff<.01,'Fresh construction exceeded INR 0.01'
    fullstate=adapter.fit_state(train,y);t=time.time();fullp=adapter.infer(fresh,fullstate,test);full_seconds=time.time()-t
    if torch.cuda.is_available():peak=int(torch.cuda.max_memory_allocated())
    else:
        import mlx.core as mx
        peak=int(mx.get_peak_memory())
    adapter.release(fresh)
    forecast=1.3*(35*fold_seconds+3*full_seconds)
    require(forecast)
    result={'status':'PASS','candidate':'L2-TF2','preprocessing':prep,'training_rows':len(a),'query_rows':len(b),'fold_seconds':fold_seconds,'full_fit_rows':3493,'full_query_rows':500,'full_seconds':full_seconds,'peak_GPU_bytes':peak,'same_state_difference_INR':same,'fresh_difference_INR':fresh_diff,'query_differences_INR':diffs,'query_RMSE_drifts_INR':drifts,'remaining_work_forecast_seconds_with_30pct_contingency':forecast,'total_pilot_seconds':time.time()-started,'pilot_accuracy_used_for_selection':False}
    # Pilot predictions are numerical witnesses only. They never populate screening caches.
    save(RESULTS/'pilot.json',result);status('pilot_passed');log(result)

def component(scheme):
    import torch,adapter
    train,test,c,partitions=dataset();y=c['y'];p=np.zeros(len(y));seen=np.zeros(len(y),int)
    frozen=identities();record=read(ROOT/'resolved.json');estimate=1.3*read(RESULTS/'pilot.json')['fold_seconds']
    for fold,indices in enumerate(partitions[scheme]):
        a,b=map(np.asarray,indices);key=scheme.replace(':','_')+f'_fold{fold}'
        path=RESULTS/'cache'/(key+'.npz');metadata=path.with_suffix('.json')
        context={'scheme':scheme,'fold':fold,'train_indices':digest(a.tolist()),'valid_indices':digest(b.tolist()),'training_targets':hashlib.sha256(y[a].tobytes()).hexdigest(),'frozen':frozen,'runtime':record}
        if metadata.exists():
            m=read(metadata);assert m['context']==context and sha(path)==m['sha256']
            with np.load(path,allow_pickle=False) as z:pred=z['prediction'].copy()
        else:
            require(estimate);status('fitting',scheme=scheme,fold=fold)
            model=create();s=adapter.fit_state(train.iloc[a],y[a]);t=time.time()
            try:pred=adapter.infer(model,s,train.iloc[b]);elapsed=time.time()-t
            finally:adapter.release(model)
            path.parent.mkdir(exist_ok=True);temp=path.with_name(path.stem+'.partial.npz');np.savez_compressed(temp,prediction=pred);temp.replace(path)
            save(metadata,{'context':context,'sha256':sha(path),'training_rows':len(a),'prediction_rows':len(b),'seconds':elapsed})
        assert pred.shape==b.shape and np.isfinite(pred).all();p[b]=pred;seen[b]+=1
        log({'scheme':scheme,'fold_complete':fold+1})
    assert (seen==1).all();return p

def screening():
    assert read(RESULTS/'pilot.json')['status']=='PASS';protocol=read(ROOT/'protocol.json')
    train,test,c,_=dataset();y=c['y'];pred={s:component(s) for s in protocol['development']};rows=[]
    for w in protocol['weights']:
        row={'weight':w};gains=[]
        for s in protocol['development']:
            base=c['v14_'+s];new=(1-w)*base+w*pred[s];old=rmse(y,base);score=rmse(y,new);gain=(old-score)/old
            row.update({s+'_baseline_RMSE':old,s+'_candidate_RMSE':score,s+'_gain_INR':old-score,s+'_relative_gain':gain});gains.append(gain)
        row['mean_relative_gain']=float(np.mean(gains));row['qualifies']=bool(min(gains)>0 and np.mean(gains)>=protocol['minimum_mean_relative_gain']);rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS/'screening.csv',index=False)
    good=sorted([r for r in rows if r['qualifies']],key=lambda r:(-r['mean_relative_gain'],r['weight']))
    if not good:
        save(RESULTS/'decision.json',{'complete':True,'qualified':False,'reason':'No development blend met the frozen gate','best':max(rows,key=lambda r:r['mean_relative_gain']),'portal_uploads':0,'retain':'V14'})
        status('complete_no_qualifier')
    else:
        save(RESULTS/'candidate_lock.json',{'weight':good[0]['weight'],'screening_sha256':sha(RESULTS/'screening.csv'),'protocol_sha256':sha(ROOT/'protocol.json'),'locked_epoch':time.time()});status('candidate_locked')
    log({'development':rows,'advanced':bool(good)})

def confirmation():
    lock=read(RESULTS/'candidate_lock.json');assert lock['screening_sha256']==sha(RESULTS/'screening.csv')
    protocol=read(ROOT/'protocol.json');train,test,c,_=dataset();y=c['y'];w=lock['weight'];rows=[];new={};base={}
    for s in protocol['confirmation']:
        p=component(s);base[s]=c['v14_'+s.replace(':','_')];new[s]=(1-w)*base[s]+w*p
        old,score=rmse(y,base[s]),rmse(y,new[s]);rows.append({'scheme':s,'v14_RMSE':old,'candidate_RMSE':score,'gain_INR':old-score,'relative_gain':(old-score)/old})
    pd.DataFrame(rows).to_csv(RESULTS/'confirmation.csv',index=False)
    rng=np.random.default_rng(protocol['bootstrap_seed']);boot=[];random=protocol['confirmation'][:3]
    for _ in range(protocol['bootstrap_resamples']):
        ix=rng.integers(0,len(y),len(y));boot.append(np.mean([rmse(y[ix],base[s][ix])-rmse(y[ix],new[s][ix]) for s in random]))
    ci=np.quantile(boot,[.025,.975]);mean=float(np.mean([r['relative_gain'] for r in rows[:3]]))
    ok=bool(all(r['gain_INR']>0 for r in rows[:3]) and mean>=.002 and ci[0]>0 and all(r['relative_gain']>=-.01 for r in rows[3:]))
    save(RESULTS/'decision.json',{'complete':True,'qualified':ok,'weight':w,'confirmation':rows,'mean_random_relative_gain':mean,'conditional_CI95_INR':ci.tolist(),'portal_uploads':0,'retain':'V14 until lower actual portal score','fresh_final_notebook_required':ok});status('confirmation_complete',qualified=ok)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['resolve','pilot','screen','confirm','full']);args=parser.parse_args()
    status('starting_'+args.stage)
    try:
        if args.stage=='resolve':resolve()
        elif args.stage=='pilot':pilot()
        elif args.stage=='screen':screening()
        elif args.stage=='confirm':confirmation()
        else:raise RuntimeError('Final export requires a qualified decision and separately verified baseline execution')
    except Exception as error:
        save(RESULTS/(args.stage+'_failure.json'),{'stage':args.stage,'type':type(error).__name__,'message':str(error)[:2000],'traceback':traceback.format_exc(),'epoch':time.time()})
        status('failed_'+args.stage,error_type=type(error).__name__)
        raise
if __name__=='__main__':main()
