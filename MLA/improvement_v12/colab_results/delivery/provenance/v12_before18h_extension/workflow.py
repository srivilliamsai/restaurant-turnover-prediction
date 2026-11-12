"""Audited V12 stages. Helpers are defined before the actual CLI entrypoint."""
import argparse,json,pickle,shutil,time,hashlib,subprocess,sys,datetime
from pathlib import Path
import numpy as np
import pandas as pd
import features,modeling,verify
from core import ROOT,STEM,DEVELOPMENT,CONFIRM,Engine,sha,digest,save,log,rmse,validate_csv

def audit(e):
    with e.cache.charge('audit'):
        for name,want in e.manifest['reference_files'].items():assert sha(ROOT/name)==want,name
        verify.feature_checks(e)
        registry=json.loads((ROOT/'baseline_registry.json').read_text())
        assert registry['inputs']==e.inputs and registry['row_ids_sha256']==e.row_hash
        for name,want in registry['generating_sources'].items():assert sha(ROOT/'provenance'/name)==want,name
        for scheme in DEVELOPMENT+['strat:27421','city']:
            rec=registry['schemes'][str(scheme)];assert sha(ROOT/rec['file'])==rec['sha256']
            assert rec['fold_sha256']==digest([b.tolist() for _,b in e.splits(scheme)])
            assert sha(ROOT/rec['v7_witness_file'])==rec['v7_witness_sha256']
            p=np.zeros(len(e.y))
            for f,(a,b) in enumerate(e.splits(scheme)):
                record=rec['tabpfn_fold_witnesses'][f];path=ROOT/'provenance/baseline_cache'/f'BASE_{str(scheme).replace(":","_")}_fold{f}.npz'
                assert sha(path)==record['sha256'] and record['context']['sources']==registry['generating_sources']
                assert record['context']['inputs']==e.inputs
                assert record['context']['train_indices']==digest(a.tolist()) and record['context']['valid_indices']==digest(b.tolist())
                assert record['context']['training_targets']==hashlib.sha256(e.y[a].tobytes()).hexdigest()
                assert record['context']['execution_contract']==e.manifest['execution_contract']
                with np.load(path,allow_pickle=False) as z:p[b]=z['prediction']
            with np.load(ROOT/rec['file'],allow_pickle=False) as z:
                assert np.array_equal(p,z['t9']) and np.array_equal(.5*z['v7']+.5*z['t9'],z['v9'])
        plan={'settings':features.SPECS,'weights':modeling.WEIGHTS,'development':DEVELOPMENT,'confirmation':CONFIRM,
              'formulas':{'T':'.5*V7+.5*((1-w)*T9+w*Tnew)','R':'(1-w)*V9+w*R'},
              'minimum_mean_relative_gain':.002,'bootstrap_lower_bound_above_zero':True,
              'max_finalists_per_family':1,'max_finalists':2,'maximum_hours':12,'sources':e.sources,'inputs':e.inputs}
        path=e.out/'experiment_plan.json'
        if path.exists():assert json.loads(path.read_text())==plan,'Frozen plan changed'
        save(path,plan);save(e.out/'audit.json',{'status':'PASS','sources':e.sources,'inputs':e.inputs,'registry_sha256':sha(ROOT/'baseline_registry.json'),'original_targets':len(e.y),'test_rows':500})
    log('AUDIT PASS: four fixed settings, sixteen blends, all targets retained.')

def pilot_context(e,name):
    a,b=e.splits(42)[0]
    return {'setting':name,'sources':e.sources,'inputs':e.inputs,
            'model_contract_sha256':sha(ROOT/'resolved_models.json'),
            'training_rows_sha256':digest(a.tolist()),'validation_rows_sha256':digest(b.tolist()),
            'training_targets_sha256':hashlib.sha256(e.y[a].tobytes()).hexdigest(),
            'checks':'original V12: full-fold fit/predict, real inference invariants, fresh repeated fit, INR 0.01'}

def read_pilot_checkpoint(e,name):
    path=e.out/'pilot_checkpoints'/f'{name}.json'
    if not path.exists():return None
    checkpoint=json.loads(path.read_text());assert checkpoint['context']==pilot_context(e,name),'Stale pilot checkpoint'
    assert digest(checkpoint['result'])==checkpoint['result_sha256'],'Corrupt pilot checkpoint'
    if 'migration_witness' in checkpoint:
        migration=json.loads((ROOT/'pilot_reuse_manifest.json').read_text())
        assert checkpoint['migration_witness']==sha(ROOT/'pilot_reuse_manifest.json')
        witness=ROOT/migration['witness_directory']
        assert sha(witness/'pilot_progress.json')==migration['original_pilot_progress_sha256']
        assert sha(witness/'resolved_models.json')==migration['original_model_contract_sha256']==sha(ROOT/'resolved_models.json')
        assert json.loads((witness/'audit.json').read_text())['sources']==migration['original_sources']
        assert migration['original_inputs']==e.inputs and migration['reuse_settings']==['T32'] and name=='T32'
        for source,want in migration['original_sources'].items():
            if source not in ['core.py','workflow.py']:assert e.sources[source]==want,'Numerical source changed during migration'
        assert checkpoint['result']==json.loads((witness/'pilot_progress.json').read_text())[name]
    return checkpoint['result']

def pilot_setting(e,name):
    """The original numerical checks, isolated so a timeout preserves other passes."""
    import torch
    config=json.loads((ROOT/'resolved_models.json').read_text());a,b=e.splits(42)[0]
    fitted=None;repeat=None
    try:
        assert all(atom in config['models'] for atom in features.SPECS[name]['atoms']),'Family unavailable in pinned runtime'
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        fitted=modeling.fit(e.train.iloc[a],e.y[a],name,log)
        p=modeling.predict(fitted,e.train.iloc[b]);seconds=time.monotonic()-start
        checks=verify.inference_checks(lambda raw:np.maximum(modeling.predict(fitted,raw),0.),e.test)
        peak=int(torch.cuda.max_memory_allocated());modeling.release(fitted);fitted=None
        changed=e.train.iloc[b].copy();changed['Registration Number']=-1;changed['Annual Turnover']=1e30
        repeat=modeling.fit(e.train.iloc[a],e.y[a],name,log);q=modeling.predict(repeat,changed)
        difference=float(np.max(np.abs(p-q)));assert difference<.01,f'Repeated fit differs INR {difference}'
        result={'status':'PASS','fold_seconds':seconds,'repeat_difference_INR':difference,'behavior':checks,'peak_gpu_allocated_bytes':peak,'training_rows':len(a),'validation_rows':len(b),'engineering_seconds':time.monotonic()-start}
    except Exception as error:
        result={'status':'unavailable','reason':{'type':type(error).__name__,'message':str(error)[:600]}}
    finally:modeling.release(fitted);modeling.release(repeat)
    save(e.out/'pilot_checkpoints'/f'{name}.json',{'context':pilot_context(e,name),'result':result,'result_sha256':digest(result)})
    return result

def pilot_setting_main():
    name,data,out=sys.argv[1:];assert name in features.SPECS
    pilot_setting(Engine(Path(data),Path(out)),name)

def pilot(e):
    records={};e.cache.stage='pilot'
    with e.cache.charge('pilot'):
        pilot_started=time.monotonic();state=e.cache.state()
        allowance=min(7200-sum(state['by_stage'].get(k,0) for k in ['audit','pilot','setup']),32400-state['seconds'],
                      datetime.datetime.fromisoformat('2026-10-29T18:00:00+05:30').timestamp()-time.time())
        path=ROOT/'resolved_models.json'
        config=json.loads(path.read_text()) if path.exists() else modeling.resolve()
        assert config['runtime']==modeling.runtime()
        assert config['checkpoint']==e.manifest['execution_contract']['checkpoint']
        # Run the cheaper family first. This changes no score-based selection.
        for name in ['R1','R4','T32','T4x16']:
            completed=read_pilot_checkpoint(e,name)
            if completed is not None:
                records[name]=completed;log({'PILOT_REUSED':name,'status':completed['status']})
            else:
                remaining=allowance-(time.monotonic()-pilot_started)-45
                if remaining<=0:
                    records[name]={'status':'unavailable','unfinished':True,'reason':{'type':'PilotTimeLimit','message':'No pilot allowance remains; completed family checks retained.'}}
                else:
                    child=subprocess.Popen([sys.executable,'-u','-c','import workflow; workflow.pilot_setting_main()',name,str(e.data),str(e.out)],cwd=ROOT)
                    try:
                        code=child.wait(timeout=remaining)
                        records[name]=read_pilot_checkpoint(e,name)
                        assert code==0 and records[name] is not None,'Pilot worker failed without a complete checkpoint'
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        records[name]={'status':'unavailable','unfinished':True,'reason':{'type':'PilotTimeLimit','message':'Setting interrupted at its remaining pilot allowance; no validation score used.'}}
                    except Exception as error:
                        records[name]={'status':'unavailable','reason':{'type':type(error).__name__,'message':str(error)[:600]}}
                    finally:
                        if child.poll() is None:child.kill();child.wait()
            log({'PILOT':name,**records[name]});save(e.out/'pilot_progress.json',records)
        # A family is admitted only if both predeclared settings pass the pilot.
        complete=[f for f in ['T','R'] if all(records[n]['status']=='PASS' for n in features.SPECS if features.SPECS[n]['family']==f)]
        def forecast(families):
            names=[n for n in features.SPECS if features.SPECS[n]['family'] in families]
            screen=sum(10*records[n]['fold_seconds'] for n in names)
            confirmation=sum(25*max(records[n]['fold_seconds'] for n in names if features.SPECS[n]['family']==f) for f in families)
            confirmation+=3600 if names else 0
            fits=screen<10800 and confirmation<18000 and screen+confirmation+e.cache.state()['seconds']+(time.monotonic()-pilot_started)+10800<=43200
            return names,screen,confirmation,fits
        # Admit complete families by measured compute cost, before any scores.
        # One expensive family must not discard another family that fits.
        families=[];budget_skips={}
        for family in sorted(complete,key=lambda f:sum(forecast([f])[1:3])):
            if forecast(families+[family])[3]:families.append(family)
            else:budget_skips[family]='Measured forecast exceeds remaining stage/total allowance with delivery reserve.'
        names,screen_forecast,confirmation_forecast,fits_budget=forecast(families)
        record={'status':'PASS' if names else 'NO_FEASIBLE_FAMILY','settings':records,'screen_settings':names,
                'screen_forecast_seconds':screen_forecast,'confirmation_forecast_seconds':confirmation_forecast,'forecast_fits_budget':fits_budget,
                'budget_skipped_families':budget_skips,'admission_order':'Lower measured compute cost; no validation RMSE used.',
                'selection_uses_validation_rmse':False,'sources':e.sources,'inputs':e.inputs,
                'model_contract_sha256':sha(ROOT/'resolved_models.json'),'model_contract':config}
        save(e.out/'pilot.json',record)

def rank(row):return (row['mean_rmse'],row['weight'],row['inference_cost'],row['name'])

def screen(e):
    pilot=json.loads((e.out/'pilot.json').read_text());assert pilot['sources']==e.sources and pilot['inputs']==e.inputs
    assert pilot['model_contract_sha256']==sha(ROOT/'resolved_models.json')
    rows=[];segments=[];folds=[];concentration=[];components=[];e.cache.stage='screen'
    refs={s:e.reference_parts(s) for s in DEVELOPMENT}
    for name in pilot['screen_settings']:
        cp={s:e.component(name,s) for s in DEVELOPMENT}
        for scheme in DEVELOPMENT:
            v7,t9=refs[scheme];base=.5*v7+.5*t9
            components.append({'name':name,'scheme':scheme,'component_rmse':rmse(e.y,np.maximum(cp[scheme],0.)),
                'v9_rmse':rmse(e.y,base),'error_correlation':float(np.corrcoef(e.y-base,e.y-cp[scheme])[0,1])})
        for weight in modeling.WEIGHTS:
            row={'name':name,'family':features.SPECS[name]['family'],'weight':weight,'inference_cost':features.SPECS[name]['cost']}
            for scheme in DEVELOPMENT:
                v7,t9=refs[scheme];base=.5*v7+.5*t9;p=modeling.combine(v7,t9,cp[scheme],name,weight);old=rmse(e.y,base)
                row[f'rmse_{scheme}']=rmse(e.y,p);row[f'gain_{scheme}']=old-row[f'rmse_{scheme}'];row[f'relative_gain_{scheme}']=row[f'gain_{scheme}']/old
                seg,fold,conc=diagnostics(e,scheme,base,p,f'{name}@{weight}');segments.extend(seg);folds.extend(fold);concentration.append(conc)
            row['mean_rmse']=float(np.mean([row[f'rmse_{s}'] for s in DEVELOPMENT]));row['mean_relative_gain']=float(np.mean([row[f'relative_gain_{s}'] for s in DEVELOPMENT]));rows.append(row)
        for filename,data in [('screening.csv',rows),('component_diagnostics.csv',components),('development_segments.csv',segments),('development_folds.csv',folds)]:pd.DataFrame(data).to_csv(e.out/filename,index=False)
        save(e.out/'development_concentration.json',concentration);save(e.out/'screening_progress.json',{'settings_complete':len(rows)//4,'planned':len(pilot['screen_settings']),'configurations':len(rows)})
    eligible=[r for r in rows if r['gain_42']>0 and r['gain_71']>0 and r['mean_relative_gain']>=.002]
    choices=[]
    for family in ['T','R']:
        selected=sorted([r for r in eligible if r['family']==family],key=rank)
        if selected:choices.append(selected[0])
    choices.sort(key=rank)
    lock={'complete':True,'choices':choices,'settings':len(pilot['screen_settings']),'configurations':len(rows),'sources':e.sources,'inputs':e.inputs,
          'model_contract_sha256':sha(ROOT/'resolved_models.json'),'experiment_plan_sha256':sha(e.out/'experiment_plan.json'),'frozen_before_confirmation':True}
    path=e.out/'candidate_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock,'Frozen shortlist changed'
    save(path,lock);log({'SCREEN_COMPLETE':True,'choices':choices})

def merge_mac(e):
    path=e.out/'mac_compute_receipt.json'
    if path.exists():
        receipt=json.loads(path.read_text());state=e.cache.state();key=digest(receipt)
        if key not in state.get('mac_receipts',[]):
            assert receipt['lock_sha256']==sha(e.out/'candidate_lock.json')
            state['seconds']+=receipt['seconds'];state['by_stage']['confirm']=state['by_stage'].get('confirm',0)+receipt['seconds'];state.setdefault('mac_receipts',[]).append(key);save(e.cache.ledger,state)

def confirm(e):
    lock=json.loads((e.out/'candidate_lock.json').read_text());assert lock['sources']==e.sources and lock['inputs']==e.inputs
    assert lock['model_contract_sha256']==sha(ROOT/'resolved_models.json');e.cache.stage='confirm';merge_mac(e)
    if not lock['choices']:
        reason='retain V9; no material development improvement' if lock['settings'] else 'retain V9; no family completed feasibility within the available budget'
        save(e.out/'decision.json',{'complete':True,'qualified':False,'choices':[],'finalists':[],'decision':reason,'portal_uploads':0});return
    refs={}
    for scheme in CONFIRM:
        try:refs[scheme]=e.reference_parts(scheme)
        except FileNotFoundError:
            save(e.out/'confirmation_progress.json',{'status':'waiting_original_mac_controls','scheme':str(scheme)});return
    finalists=[];segments=[];folds=[];concentration=[]
    for choice in lock['choices']:
        metrics={};bases={};predictions={}
        for scheme in CONFIRM:
            v7,t9=refs[scheme];base=.5*v7+.5*t9;c=e.component(choice['name'],scheme);p=modeling.combine(v7,t9,c,choice['name'],choice['weight'])
            old,new=rmse(e.y,base),rmse(e.y,p);metrics[str(scheme)]={'v9_rmse':old,'candidate_rmse':new,'gain':old-new,'relative_gain':1-new/old}
            bases[scheme]=base;predictions[scheme]=p
            np.savez_compressed(e.out/f"confirmation_{choice['name']}_{str(scheme).replace(':','_')}.npz",baseline=base,candidate=p)
            seg,fold,conc=diagnostics(e,scheme,base,p,choice['name']);segments.extend(seg);folds.extend(fold);concentration.append(conc)
            save(e.out/'confirmation_progress.json',{'status':'running','name':choice['name'],'metrics':metrics});log({'CONFIRMATION':choice['name'],'scheme':str(scheme),**metrics[str(scheme)]})
        interval=uncertainty(e.y,bases,predictions);mean=float(np.mean([metrics[str(s)]['relative_gain'] for s in CONFIRM[:3]]))
        qualified=all(metrics[str(s)]['gain']>0 for s in CONFIRM[:3]) and mean>=.002 and interval['ci95_INR'][0]>0 and all(metrics[s]['relative_gain']>=-.01 for s in ['strat:27421','city'])
        finalists.append({'choice':choice,'qualified':bool(qualified),'metrics':metrics,'uncertainty':interval,'mean_random_relative_gain':mean})
    for filename,data in [('confirmation_segments.csv',segments),('confirmation_folds.csv',folds)]:pd.DataFrame(data).to_csv(e.out/filename,index=False)
    pd.DataFrame([{'name':f['choice']['name'],'scheme':s,**m} for f in finalists for s,m in f['metrics'].items()]).to_csv(e.out/'confirmation.csv',index=False)
    save(e.out/'confirmation_concentration.json',concentration)
    choices=[f['choice'] for f in finalists if f['qualified']]
    decision={'complete':True,'qualified':bool(choices),'choices':choices,'finalists':finalists,'lock_sha256':sha(e.out/'candidate_lock.json'),
              'decision':f'{len(choices)} independently qualified V12 portal trial(s)' if choices else 'retain V9; finalists failed confirmation gates','portal_uploads':0}
    merge_mac(e);save(e.out/'decision.json',decision);save(e.out/'confirmation_progress.json',{'status':'complete','qualified':bool(choices)});log(decision)

def export(e):
    import baseline,build_delivery
    decision=json.loads((e.out/'decision.json').read_text());assert decision['complete'];target=ROOT/'delivery';target.mkdir(exist_ok=True)
    with e.cache.charge('export'):
        if decision['qualified']:
            with (ROOT/'reference/verified_v7.pkl').open('rb') as f:original=pickle.load(f)
            b=baseline.predict(original,e.test);assert np.max(np.abs(b-pd.read_csv(ROOT/'reference/scored_v7.csv')['Annual Turnover'].to_numpy(float)))<.01
            fit=modeling.fit_atomic(e.train,e.y,'BASE',log);t=modeling.predict_atomic(fit,e.test);modeling.release(fit);v9=.5*b+.5*t
            diff=float(np.max(np.abs(v9-pd.read_csv(ROOT/'reference/scored_v9.csv')['Annual Turnover'].to_numpy(float))));assert diff<.01
            for number,choice in enumerate(decision['choices'],1):
                directory=target/f"trial_{number}_{choice['name']}";directory.mkdir(exist_ok=True)
                fit=modeling.fit(e.train,e.y,choice['name'],log)
                predict=lambda raw:np.maximum(modeling.predict(fit,raw),0.)
                behavior=verify.inference_checks(predict,e.test);c=modeling.predict(fit,e.test);modeling.release(fit)
                p=modeling.combine(b,t,c,choice['name'],choice['weight']);checks=verify.csv_checks(e,p)
                frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p});validate_csv(frame,e.test);frame.to_csv(directory/(STEM+'.csv'),index=False)
                save(directory/'export_checks.json',{'status':'PASS','choice':choice,'fresh_v9_difference_INR':diff,'behavior':behavior,'csv_checks':checks,'csv_sha256':sha(directory/(STEM+'.csv'))})
                build_delivery.build_and_execute(e,{'qualified':True,'choice':choice},directory)
        else:
            for suffix in ['.csv','.ipynb']:shutil.copy2(ROOT/('reference/scored_v9'+suffix),target/('KEEP_V9_'+STEM+suffix))
        build_delivery.package(e,decision,target)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','pilot','screen','confirm','export']);parser.add_argument('--data-dir',type=Path);parser.add_argument('--output-dir',type=Path);parser.add_argument('--budget-hours',type=float,default=12);args=parser.parse_args()
    e=Engine(args.data_dir,args.output_dir,args.budget_hours);e.cache.stage=args.stage;save(e.out/'stage_status.json',{'stage':args.stage,'status':'running'})
    try:
        globals()[args.stage](e);waiting=args.stage=='confirm' and not (e.out/'decision.json').exists()
        save(e.out/'stage_status.json',{'stage':args.stage,'status':'waiting_mac_controls' if waiting else 'complete'})
    except Exception as error:
        save(e.out/'stage_status.json',{'stage':args.stage,'status':'failed','error_type':type(error).__name__,'error':str(error)[:500]});raise

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



if __name__=='__main__':main()
