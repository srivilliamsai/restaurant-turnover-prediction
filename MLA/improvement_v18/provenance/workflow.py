"""Auditable bounded V10 stages: six feature settings, 24 fixed replacement blends."""
import argparse
import importlib.metadata
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import features
import modeling
import verify
from core import (ROOT,STEM,DEVELOPMENT,CONFIRM,PINNED,Engine,sha,digest,save,log,rmse,validate_csv,BudgetExceeded)


def audit(e):
    with e.cache.charge('audit'):
        for name,expected in e.manifest['reference_files'].items():assert sha(ROOT/name)==expected,name
        verify.feature_checks(e)
        # Audit preserved references without launching GPU control refits.
        for scheme in DEVELOPMENT+[48211,48212,48213,'strat:27421']:
            record=e.manifest['oof'][str(scheme).replace(':','_')]
            assert sha(ROOT/record['file'])==record['sha256']
            assert record['fold_sha256']==digest([b.tolist() for _,b in e.splits(scheme)])
        settings={'settings':features.SPECS,'feature_counts':features.COUNTS,'replacement_weights':modeling.WEIGHTS,
            'formula':'0.5*V7 + 0.5*((1-w)*T9+w*Tnew)', 'development':DEVELOPMENT,'confirmation':CONFIRM,
            'seed':10242,'estimators':16,'target':'raw INR millions','checkpoint':e.manifest['execution_contract'],
            'inference':'Fixed 500-row internal batches; frozen training-feature padding; discard padded outputs; no output rounding',
            'ranking':['mean_rmse','replacement_weight','feature_count','name'],
            'max_finalists':1,'max_settings':6,'max_configurations':24,
            'qualification':'all three random gains positive; stratified/city deterioration <=1%',
            'budget_hours':e.cache.hours,'sources':e.sources,'inputs':e.inputs}
        path=e.out/'experiment_plan.json'
        if path.exists():assert json.loads(path.read_text())==settings,'Frozen experiment plan changed.'
        save(path,settings)
        save(e.out/'audit.json',{'status':'PASS','inputs':e.inputs,'sources':e.sources,'train_rows':len(e.y),
            'test_rows':len(e.test),'v9_portal_rmse':e.manifest['v9_portal_rmse'],'original_targets_retained':True,
            'source_inventory_sha256':sha(ROOT/'source_inventory.json'),'scored_v9_sha256':sha(ROOT/'reference/scored_v9.csv'),
            'scored_v9_notebook_sha256':sha(ROOT/'reference/scored_v9.ipynb'),
            'revised_city_fold_sha256':digest([b.tolist() for _,b in e.splits('city')]),'paid_resources_requested':False})
    log('AUDIT PASS: six settings, 24 selectable blends; original targets and scored V9 preserved.')


def pilot(e):
    import torch
    with e.cache.charge('pilot'):
        runtime={name:importlib.metadata.version(name) for name in PINNED}
        assert runtime==PINNED,f'Pinned dependencies differ: {runtime}'
        contract=modeling.execution_contract()
        a,b=e.splits(42)[0]
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        fitted=modeling.fit(e.train.iloc[a],e.y[a],'C+S+G',log)
        p=modeling.predict(fitted,e.train.iloc[b])
        checks=verify.inference_checks(lambda raw:modeling.predict(fitted,raw),e.test)
        frame=fitted['builder'].transform(e.train.iloc[b])
        changed=e.train.iloc[b].copy();changed['Annual Turnover']=1e30
        pd.testing.assert_frame_equal(frame,fitted['builder'].transform(changed))
        q=modeling.fit_predict(e.train.iloc[a],e.y[a],changed,'C+S+G',log)
        difference=float(np.max(np.abs(q-p)));assert difference<.01
        record={'status':'PASS','inputs':e.inputs,'sources':e.sources,'execution_contract':contract,
            'runtime':runtime,'training_rows':len(a),'validation_rows':len(b),'largest_features':66,
            'repeat_difference_INR':difference,'behavior':checks,'seconds':time.monotonic()-started,
            'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated()),
            'outer_validation_target_changes_leave_predictions_unchanged':True}
        save(e.out/'pilot.json',record)
    log({'PILOT':'PASS','seconds':record['seconds'],'repeat_difference_INR':difference})


def blend(b,t,c,w):return .5*b+.5*((1-w)*t+w*c)


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


def screen(e):
    pilot_record=json.loads((e.out/'pilot.json').read_text())
    assert pilot_record['status']=='PASS' and pilot_record['sources']==e.sources
    bases={s:e.reference(s) for s in DEVELOPMENT}
    rows=[];components=[];segments=[];folds=[];concentration=[]
    e.cache.stage='screen'
    for name in features.SPECS:
        cp={s:e.component(name,s) for s in DEVELOPMENT}
        for s in DEVELOPMENT:
            components.append({'name':name,'scheme':s,'component_rmse':rmse(e.y,cp[s])})
        for w in modeling.WEIGHTS:
            row={'name':name,'replacement_weight':w,'feature_count':features.COUNTS[name]}
            for s in DEVELOPMENT:
                b,t,base=bases[s];p=blend(b,t,cp[s],w)
                row[f'rmse_{s}']=rmse(e.y,p);row[f'gain_{s}']=rmse(e.y,base)-row[f'rmse_{s}']
                seg,fold,conc=diagnostics(e,s,base,p,name+f'@{w}')
                segments.extend(seg);folds.extend(fold);concentration.append(conc)
            row['mean_rmse']=float(np.mean([row[f'rmse_{s}'] for s in DEVELOPMENT]));rows.append(row)
        pd.DataFrame(rows).to_csv(e.out/'screening.csv',index=False)
        pd.DataFrame(components).to_csv(e.out/'component_diagnostics.csv',index=False)
        pd.DataFrame(segments).to_csv(e.out/'development_segments.csv',index=False)
        pd.DataFrame(folds).to_csv(e.out/'development_folds.csv',index=False)
        save(e.out/'development_concentration.json',concentration)
        save(e.out/'screening_progress.json',{'settings_complete':len(rows)//4,'planned':6,'configurations':len(rows)})
    table=pd.DataFrame(rows).sort_values(['mean_rmse','replacement_weight','feature_count','name'])
    eligible=table[(table.gain_42>0)&(table.gain_71>0)]
    choice=eligible.iloc[0].to_dict() if len(eligible) else None
    if choice:choice['feature_count']=int(choice['feature_count'])
    lock={'choice':choice,'complete':True,'settings':6,'configurations':24,'sources':e.sources,'inputs':e.inputs,
          'plan_sha256':sha(e.out/'experiment_plan.json'),'frozen_before_confirmation':True}
    path=e.out/'candidate_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock,'Frozen finalist changed.'
    save(path,lock);save(e.out/'screening_decision.json',lock)
    log({'SCREENING_COMPLETE':True,'qualified_for_confirmation':choice})


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


def confirm(e):
    lock=json.loads((e.out/'candidate_lock.json').read_text())
    assert lock['complete'] and lock['sources']==e.sources and lock['inputs']==e.inputs
    choice=lock['choice']
    if choice is None:
        decision={'complete':True,'qualified':False,'choice':None,'decision':'retain scored V9; no development blend improved both assignments',
                  'lock_sha256':sha(e.out/'candidate_lock.json'),'portal_uploads':0}
        save(e.out/'decision.json',decision);log(decision);return
    e.cache.stage='confirm';metrics={};bases={};predictions={};segments=[];folds=[];concentration=[]
    for scheme in CONFIRM:
        try:b,t,base=e.reference(scheme)
        except FileNotFoundError:
            save(e.out/'confirmation_progress.json',{'status':'waiting_original_platform_city_baseline','metrics':metrics})
            log('Confirmation random/stratified results saved; waiting for exact revised-city Mac baseline.');return
        c=e.component(choice['name'],scheme);p=blend(b,t,c,choice['replacement_weight'])
        old,new=rmse(e.y,base),rmse(e.y,p)
        metrics[str(scheme)]={'v9_rmse':old,'candidate_rmse':new,'gain':old-new,'relative_change':new/old-1}
        bases[scheme]=base;predictions[scheme]=p
        np.savez_compressed(e.out/(f'confirmation_{str(scheme).replace(":","_")}.npz'),baseline=base,candidate=p)
        seg,fold,conc=diagnostics(e,scheme,base,p,choice['name'])
        segments.extend(seg);folds.extend(fold);concentration.append(conc)
        pd.DataFrame([{'scheme':s,**m} for s,m in metrics.items()]).to_csv(e.out/'confirmation.csv',index=False)
        pd.DataFrame(segments).to_csv(e.out/'confirmation_segments.csv',index=False)
        pd.DataFrame(folds).to_csv(e.out/'confirmation_folds.csv',index=False)
        save(e.out/'confirmation_concentration.json',concentration)
        save(e.out/'confirmation_progress.json',{'status':'running','metrics':metrics})
        log({'CONFIRMATION':str(scheme),**metrics[str(scheme)]})
    qualified=all(metrics[str(s)]['gain']>0 for s in CONFIRM[:3]) and all(metrics[s]['relative_change']<=.01 for s in ['strat:27421','city'])
    decision={'complete':True,'qualified':bool(qualified),'choice':choice,'metrics':metrics,
        'uncertainty':uncertainty(e.y,bases,predictions),'lock_sha256':sha(e.out/'candidate_lock.json'),
        'decision':'one qualified V10 portal trial' if qualified else 'retain scored V9; finalist failed confirmation',
        'mean_random_gain':float(np.mean([metrics[str(s)]['gain'] for s in CONFIRM[:3]])),
        'baseline_portal_rmse':11904515.9864996,'portal_uploads':0}
    save(e.out/'decision.json',decision);save(e.out/'confirmation_progress.json',{'status':'complete','qualified':bool(qualified)})
    log({'DECISION':decision['decision']})


def export(e):
    import baseline
    decision=json.loads((e.out/'decision.json').read_text());assert decision['complete']
    with e.cache.charge('export'):
        target=ROOT/'delivery';target.mkdir(exist_ok=True)
        model_path=ROOT/'reference/verified_v7.pkl'
        assert sha(model_path)==e.manifest['v7_checkpoint_sha256']
        with model_path.open('rb') as file:fitted_base=pickle.load(file)
        b=baseline.predict(fitted_base,e.test)
        expected_b=pd.read_csv(ROOT/'reference/scored_v7.csv')['Annual Turnover'].to_numpy(float)
        base_diff=float(np.max(np.abs(b-expected_b)));assert base_diff<.01
        base_checks=verify.inference_checks(lambda raw:baseline.predict(fitted_base,raw),e.test)
        fitted_t9=modeling.fit(e.train,e.y,'BASE',log);t=modeling.predict(fitted_t9,e.test)
        v9=.5*b+.5*t
        expected_v9=pd.read_csv(ROOT/'reference/scored_v9.csv')['Annual Turnover'].to_numpy(float)
        v9_diff=float(np.max(np.abs(v9-expected_v9)));assert v9_diff<.01
        t9_checks=verify.inference_checks(lambda raw:modeling.predict(fitted_t9,raw),e.test)
        record={'v7_portable_difference_INR':base_diff,'fresh_v9_difference_INR':v9_diff,
                'v7_retrained_on_linux':False,'t9_freshly_fitted':True,'v7_behavior':base_checks,'t9_behavior':t9_checks}
        if decision['qualified']:
            choice=decision['choice'];fitted=modeling.fit(e.train,e.y,choice['name'],log)
            c=modeling.predict(fitted,e.test);p=blend(b,t,c,choice['replacement_weight'])
            record['candidate_behavior']=verify.inference_checks(lambda raw:modeling.predict(fitted,raw),e.test,features.SPECS[choice['name']]['prune'])
            record['candidate_feature_map_sha256']=digest(fitted['builder'].state())
            record['csv_checks']=verify.csv_checks(e,p)
            frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p})
            validate_csv(frame,e.test);frame.to_csv(target/(STEM+'.csv'),index=False)
            record['candidate_csv_sha256']=sha(target/(STEM+'.csv'))
        else:
            shutil.copy2(ROOT/'reference/scored_v9.csv',target/('KEEP_V9_'+STEM+'.csv'))
            shutil.copy2(ROOT/'reference/scored_v9.ipynb',target/('KEEP_V9_'+STEM+'.ipynb'))
            record['csv_checks']=verify.csv_checks(e,v9)
        record.update(status='PASS',decision=decision,inputs=e.inputs,sources=e.sources,execution_contract=modeling.execution_contract())
        save(target/'export_checks.json',record)
        # Build and freshly execute the standalone reproduction notebook in a
        # separate interpreter; it never uses expected predictions as features.
        import build_delivery
        build_delivery.build_and_execute(e,decision,target)
    log({'EXPORT':'PASS','qualified':decision['qualified'],'delivery':str(target)})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['audit','pilot','screen','confirm','export','all'])
    parser.add_argument('--data-dir',default=str(ROOT.parent))
    parser.add_argument('--output-dir',default=str(ROOT/'results'))
    parser.add_argument('--budget-hours',type=float,default=18)
    args=parser.parse_args();e=Engine(args.data_dir,args.output_dir,args.budget_hours)
    stages=['audit','pilot','screen','confirm','export'] if args.stage=='all' else [args.stage]
    try:
        for stage in stages:
            e.cache.stage=stage;save(e.out/'stage_status.json',{'stage':stage,'status':'running'})
            globals()[stage](e)
            pending=stage=='confirm' and not (e.out/'decision.json').exists()
            save(e.out/'stage_status.json',{'stage':stage,'status':'waiting_city_baseline' if pending else 'complete'})
            if pending:break
    except Exception as error:
        save(e.out/'stage_status.json',{'stage':e.cache.stage,'status':'failed','error':repr(error)})
        raise


if __name__=='__main__':main()
