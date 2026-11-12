"""V11 measured stages; stricter predeclared improvement and uncertainty gates."""
import argparse,json,pickle,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import features,modeling,verify
from core import ROOT,STEM,DEVELOPMENT,CONFIRM,PINNED,Engine,sha,digest,save,log,rmse,validate_csv

def audit(e):
    with e.cache.charge('audit'):
        for name,want in e.manifest['reference_files'].items():
            p=ROOT/name
            assert p.exists() and sha(p)==want,name
        verify.feature_checks(e)
        registry=json.loads((ROOT/'baseline_registry.json').read_text())
        assert registry['inputs']==e.inputs and registry['row_ids_sha256']==e.row_hash
        for name,want in registry['generating_sources'].items():assert sha(ROOT/'provenance'/name)==want,name
        for scheme in DEVELOPMENT+['strat:27421','city']:
            rec=registry['schemes'][str(scheme)];assert sha(ROOT/rec['file'])==rec['sha256']
            assert rec['fold_sha256']==digest([b.tolist() for _,b in e.splits(scheme)])
            assert sha(ROOT/rec['v7_witness_file'])==rec['v7_witness_sha256']
            combined=np.zeros(len(e.y))
            for f,(_,indices) in enumerate(e.splits(scheme)):
                meta=rec['tabpfn_fold_witnesses'][f];path=ROOT/'provenance/baseline_cache'/f'BASE_{str(scheme).replace(":","_")}_fold{f}.npz'
                assert sha(path)==meta['sha256'] and meta['context']['sources']==registry['generating_sources'] and meta['context']['inputs']==e.inputs
                with np.load(path,allow_pickle=False) as z:combined[indices]=z['prediction']
            with np.load(ROOT/rec['file'],allow_pickle=False) as z:
                assert np.array_equal(combined,z['t9']) and np.array_equal(.5*z['v7']+.5*z['t9'],z['v9'])
        plan={'settings':features.SPECS,'weights':modeling.WEIGHTS,'formula':'(1-w)*V9+w*M','development':DEVELOPMENT,'confirmation':CONFIRM,
              'minimum_mean_relative_gain':.002,'bootstrap_lower_bound_above_zero':True,'max_finalists':1,'maximum_hours':8,'sources':e.sources,'inputs':e.inputs}
        p=e.out/'experiment_plan.json'
        if p.exists():assert json.loads(p.read_text())==plan,'Frozen plan changed'
        save(p,plan);save(e.out/'audit.json',{'status':'PASS','sources':e.sources,'inputs':e.inputs,'verified_baseline_registry_sha256':sha(ROOT/'baseline_registry.json'),'original_targets':len(e.y),'test_rows':500})
    log('AUDIT PASS: four fixed settings, sixteen blends; V9 and every target preserved.')

def pilot(e):
    import torch
    e.cache.stage='pilot';records={};config=None
    with e.cache.charge('pilot'):
        config=modeling.resolve();a,b=e.splits(42)[0]
        for family in ['I','N']:
            if family not in config['models']:continue
            while True:
                fitted=None;repeat=None
                try:
                    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                    fitted=modeling.fit(e.train.iloc[a],e.y[a],family+'16',log)
                    p=modeling.predict(fitted,e.train.iloc[b]);fold_seconds=time.monotonic()-start
                    checks=verify.inference_checks(lambda raw:modeling.predict(fitted,raw),e.test)
                    peak=int(torch.cuda.max_memory_allocated());modeling.release(fitted);fitted=None
                    changed=e.train.iloc[b].copy();changed['Registration Number']=-1;changed['Annual Turnover']=1e30
                    repeat=modeling.fit(e.train.iloc[a],e.y[a],family+'16',log)
                    q=modeling.predict(repeat,changed);difference=float(np.max(np.abs(p-q)));assert difference<.01,f'Repeated model differs INR {difference}'
                    records[family]={'status':'PASS','training_rows':len(a),'validation_rows':len(b),'fold_seconds':fold_seconds,'repeat_difference_INR':difference,'behavior':checks,'peak_gpu_allocated_bytes':peak,'engineering_seconds':time.monotonic()-start}
                    log({'PILOT_FAMILY':family,**records[family]});break
                except Exception as error:
                    reason={'error_type':type(error).__name__,'message':str(error)[:500]}
                    log({'PILOT_ENGINEERING_FAILURE':family,**reason})
                    if family=='N' and config['models']['N']['checkpoint']['version']=='3.5':
                        modeling.release(fitted);fitted=None;modeling.release(repeat);repeat=None
                        config=modeling.fallback(config,reason);continue
                    records[family]={'status':'unavailable','reason':reason};config['models'].pop(family,None);save(ROOT/'resolved_models.json',config);break
                finally:
                    modeling.release(fitted);modeling.release(repeat)
        feasible=[f for f in ['I','N'] if records.get(f,{}).get('status')=='PASS']
        forecasts={f:16.5*records[f]['fold_seconds'] for f in feasible}
        available_screen=7200-e.cache.state()['by_stage'].get('screen',0)
        retained=feasible.copy()
        if sum(forecasts.values())>available_screen and retained:
            retained=[min(retained,key=lambda f:(forecasts[f],f))]
        if retained and sum(forecasts[f] for f in retained)>available_screen:retained=[]
        # Engineering-only budget choice, before any development RMSE is read.
        names=[name for name in features.SPECS if name[0] in retained]
        save(ROOT/'resolved_models.json',config)
        record={'status':'PASS' if names else 'NO_FEASIBLE_FAMILY','families':records,'screen_settings':names,'screen_forecast_seconds':forecasts,
                'selection_uses_validation_rmse':False,'sources':e.sources,'inputs':e.inputs,'model_contract_sha256':sha(ROOT/'resolved_models.json'),'model_contract':config}
        save(e.out/'pilot.json',record)

def blend(base,c,w):return (1-w)*base+w*c

def screen(e):
    pilot=json.loads((e.out/'pilot.json').read_text());assert pilot['sources']==e.sources and pilot['inputs']==e.inputs
    assert pilot['model_contract_sha256']==sha(ROOT/'resolved_models.json')
    names=pilot['screen_settings'];rows=[];segments=[];folds=[];concentration=[];components=[]
    e.cache.stage='screen';bases={s:e.reference(s) for s in DEVELOPMENT}
    for name in names:
        cp={s:e.component(name,s) for s in DEVELOPMENT}
        for s in DEVELOPMENT:components.append({'name':name,'scheme':s,'component_rmse':rmse(e.y,cp[s])})
        for w in modeling.WEIGHTS:
            row={'name':name,'weight':w,'estimators':features.SPECS[name]['estimators']}
            for s in DEVELOPMENT:
                p=blend(bases[s],cp[s],w);old=rmse(e.y,bases[s]);row[f'rmse_{s}']=rmse(e.y,p);row[f'gain_{s}']=old-row[f'rmse_{s}'];row[f'relative_gain_{s}']=row[f'gain_{s}']/old
                seg,fold,conc=diagnostics(e,s,bases[s],p,name+f'@{w}');segments.extend(seg);folds.extend(fold);concentration.append(conc)
            row['mean_rmse']=float(np.mean([row[f'rmse_{s}'] for s in DEVELOPMENT]));row['mean_relative_gain']=float(np.mean([row[f'relative_gain_{s}'] for s in DEVELOPMENT]));rows.append(row)
        for filename,data in [('screening.csv',rows),('component_diagnostics.csv',components),('development_segments.csv',segments),('development_folds.csv',folds)]:pd.DataFrame(data).to_csv(e.out/filename,index=False)
        save(e.out/'development_concentration.json',concentration);save(e.out/'screening_progress.json',{'settings_complete':len(rows)//4,'planned':len(names),'configurations':len(rows)})
    eligible=[r for r in rows if r['gain_42']>0 and r['gain_71']>0 and r['mean_relative_gain']>=.002]
    eligible.sort(key=lambda r:(r['mean_rmse'],r['weight'],r['estimators'],r['name']))
    lock={'complete':True,'choice':eligible[0] if eligible else None,'settings':len(names),'configurations':len(rows),'sources':e.sources,'inputs':e.inputs,'model_contract_sha256':sha(ROOT/'resolved_models.json'),'experiment_plan_sha256':sha(e.out/'experiment_plan.json'),'frozen_before_confirmation':True}
    p=e.out/'candidate_lock.json'
    if p.exists():assert json.loads(p.read_text())==lock,'Frozen choice changed'
    save(p,lock);log({'SCREEN_COMPLETE':True,'choice':lock['choice']})

def merge_mac(e):
    p=e.out/'mac_compute_receipt.json'
    if p.exists():
        receipt=json.loads(p.read_text());s=e.cache.state();key=digest(receipt)
        if key not in s.get('mac_receipts',[]):
            s['seconds']+=receipt['seconds'];s['by_stage']['confirm']=s['by_stage'].get('confirm',0)+receipt['seconds'];s.setdefault('mac_receipts',[]).append(key);save(e.cache.ledger,s)

def confirm(e):
    lock=json.loads((e.out/'candidate_lock.json').read_text());assert lock['sources']==e.sources and lock['inputs']==e.inputs
    assert lock['model_contract_sha256']==sha(ROOT/'resolved_models.json');choice=lock['choice'];e.cache.stage='confirm';merge_mac(e)
    if choice is None:
        save(e.out/'decision.json',{'complete':True,'qualified':False,'choice':None,'decision':'retain V9; no material development improvement','portal_uploads':0});return
    metrics={};bases={};predictions={};segments=[];folds=[];concentration=[]
    for scheme in CONFIRM:
        try:base=e.reference(scheme)
        except FileNotFoundError:
            save(e.out/'confirmation_progress.json',{'status':'waiting_original_mac_controls','metrics':metrics});return
        c=e.component(choice['name'],scheme);p=blend(base,c,choice['weight']);old,new=rmse(e.y,base),rmse(e.y,p)
        metrics[str(scheme)]={'v9_rmse':old,'candidate_rmse':new,'gain':old-new,'relative_gain':1-new/old}
        bases[scheme]=base;predictions[scheme]=p;np.savez_compressed(e.out/f'confirmation_{str(scheme).replace(":","_")}.npz',baseline=base,candidate=p)
        seg,fold,conc=diagnostics(e,scheme,base,p,choice['name']);segments.extend(seg);folds.extend(fold);concentration.append(conc)
        for filename,data in [('confirmation.csv',[{'scheme':k,**v} for k,v in metrics.items()]),('confirmation_segments.csv',segments),('confirmation_folds.csv',folds)]:pd.DataFrame(data).to_csv(e.out/filename,index=False)
        save(e.out/'confirmation_concentration.json',concentration);save(e.out/'confirmation_progress.json',{'status':'running','metrics':metrics});log({'CONFIRMATION':str(scheme),**metrics[str(scheme)]})
    interval=uncertainty(e.y,bases,predictions);mean_gain=float(np.mean([metrics[str(s)]['relative_gain'] for s in CONFIRM[:3]]))
    qualified=all(metrics[str(s)]['gain']>0 for s in CONFIRM[:3]) and mean_gain>=.002 and interval['ci95_INR'][0]>0 and all(metrics[s]['relative_gain']>=-.01 for s in ['strat:27421','city'])
    decision={'complete':True,'qualified':bool(qualified),'choice':choice,'metrics':metrics,'uncertainty':interval,'mean_random_relative_gain':mean_gain,'lock_sha256':sha(e.out/'candidate_lock.json'),'decision':'one qualified V11 portal trial' if qualified else 'retain V9; finalist failed stricter confirmation gates','portal_uploads':0}
    merge_mac(e);save(e.out/'decision.json',decision);save(e.out/'confirmation_progress.json',{'status':'complete','qualified':bool(qualified)});log(decision)

def export(e):
    import baseline,build_delivery
    decision=json.loads((e.out/'decision.json').read_text());assert decision['complete']
    target=ROOT/'delivery';target.mkdir(exist_ok=True)
    with e.cache.charge('export'):
        if decision['qualified']:
            with (ROOT/'reference/verified_v7.pkl').open('rb') as f:model=pickle.load(f)
            b=baseline.predict(model,e.test);expected=pd.read_csv(ROOT/'reference/scored_v7.csv')['Annual Turnover'].to_numpy(float)
            assert np.max(np.abs(b-expected))<.01
            fitted=modeling.fit(e.train,e.y,'BASE',log);t=modeling.predict(fitted,e.test);modeling.release(fitted);v9=.5*b+.5*t
            difference=float(np.max(np.abs(v9-pd.read_csv(ROOT/'reference/scored_v9.csv')['Annual Turnover'].to_numpy(float))));assert difference<.01
            fitted=modeling.fit(e.train,e.y,decision['choice']['name'],log);c=modeling.predict(fitted,e.test)
            behavior=verify.inference_checks(lambda raw:modeling.predict(fitted,raw),e.test);modeling.release(fitted)
            p=blend(v9,c,decision['choice']['weight']);checks=verify.csv_checks(e,p)
            frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p});validate_csv(frame,e.test);frame.to_csv(target/(STEM+'.csv'),index=False)
            save(target/'export_checks.json',{'status':'PASS','fresh_v9_difference_INR':difference,'candidate_behavior':behavior,'csv_checks':checks,'csv_sha256':sha(target/(STEM+'.csv'))})
            build_delivery.build_and_execute(e,decision,target)
        else:
            for suffix in ['.csv','.ipynb']:shutil.copy2(ROOT/('reference/scored_v9'+suffix),target/('KEEP_V9_'+STEM+suffix))
        build_delivery.package(e,decision,target)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','pilot','screen','confirm','export']);parser.add_argument('--data-dir',type=Path);parser.add_argument('--output-dir',type=Path);parser.add_argument('--budget-hours',type=float,default=8);args=parser.parse_args()
    e=Engine(args.data_dir,args.output_dir,args.budget_hours);e.cache.stage=args.stage;save(e.out/'stage_status.json',{'stage':args.stage,'status':'running'})
    try:
        globals()[args.stage](e)
        waiting=args.stage=='confirm' and not (e.out/'decision.json').exists()
        save(e.out/'stage_status.json',{'stage':args.stage,'status':'waiting_mac_controls' if waiting else 'complete'})
    except Exception as error:
        save(e.out/'stage_status.json',{'stage':args.stage,'status':'failed','error_type':type(error).__name__,'error':str(error)[:500]});raise

if __name__=='__main__':main()

def diagnostics(e,scheme,base,pred,name):
    reduction=(e.y-base)**2-(e.y-pred)**2
    total=float(reduction.sum());sse=float(np.sum((e.y-pred)**2))
    segments=[];folds=[]
    for label,mask in [('<=40M',e.y<=40e6),('40-80M',(e.y>40e6)&(e.y<=80e6)),('>80M',e.y>80e6)]:
        gain=float(reduction[mask].sum())
        segments.append({'name':name,'scheme':str(scheme),'segment':label,'rows':int(mask.sum()),
            'v9_rmse':rmse(e.y[mask],base[mask]),'candidate_rmse':rmse(e.y[mask],pred[mask]),
            'candidate_sse_share':float(np.sum((e.y[mask]-pred[mask])**2)/sse),
            'sse_reduction':gain,'share_net_sse_reduction':gain/total if total else None})
    for f,(_,indices) in enumerate(e.splits(scheme)):
        folds.append({'name':name,'scheme':str(scheme),'fold':f,'rows':len(indices),
            'v9_rmse':rmse(e.y[indices],base[indices]),'candidate_rmse':rmse(e.y[indices],pred[indices]),
            'above_80M':int((e.y[indices]>80e6).sum())})
    top=np.argsort(-reduction)[:10];top=top[reduction[top]>0]
    concentration={'name':name,'scheme':str(scheme),'net_sse_reduction':total,
        'largest_ten_positive_share':float(reduction[top].sum()/total) if total else None,
        'largest_positive_rows':[{'training_row':int(i),'restaurant_id':int(e.train['Registration Number'].iloc[i]),
            'turnover':float(e.y[i]),'sse_reduction':float(reduction[i])} for i in top]}
    return segments,folds,concentration


def uncertainty(y,bases,predictions):
    rng=np.random.default_rng(8319);values=[]
    for _ in range(50):
        index=rng.integers(0,len(y),(100,len(y)))
        one=[]
        for scheme in CONFIRM[:3]:
            be=(y-bases[scheme])**2;ce=(y-predictions[scheme])**2
            one.append(np.sqrt(be[index].mean(axis=1))-np.sqrt(ce[index].mean(axis=1)))
        values.extend(np.mean(one,axis=0).tolist())
    return {'resamples':5000,'seed':8319,'ci95_INR':np.quantile(values,[.025,.975]).tolist(),
        'positive_fraction':float(np.mean(np.asarray(values)>0)),
        'interpretation':'Conditional paired restaurant rows across three random assignments; excludes selection, retraining and distribution uncertainty.'}


