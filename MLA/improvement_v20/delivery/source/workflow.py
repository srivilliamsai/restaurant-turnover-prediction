"""Finite V20 audit, three bounded pilots, fifteen comparisons and frozen confirmation."""
import argparse
import os
import hashlib
import json
import pickle
import threading
import time
import numpy as np
import pandas as pd
import features
import modeling
import verify
from core import ROOT, STEM, DEVELOPMENT, CONFIRM, Engine, sha, digest, save, log, rmse, validate_csv


def audit(e):
    verify.reference_checks(e)
    verify.feature_checks(e)
    assert sha(ROOT / 'reference/scored_v14.csv') == '39d49f79beb8549d04ffc78d134aad2407dc5e73914dc3c30e8bb162a3c0ba05'
    assert sha(ROOT / 'reference/scored_v9.csv') == '5a384dc3d8bf58c4e29f0d6cac646aa33afe22bdcdd186cccd9758342769bbd4'
    plan = {'settings': features.SPECS, 'weights': features.WEIGHTS, 'routes': features.ROUTES,
            'development': DEVELOPMENT, 'confirmation': CONFIRM,
            'formula': '(1-w)*V14+w*M', 'control': 'V14',
            'minimum_mean_relative_gain': .002,
            'maximum_stress_deterioration': .01, 'maximum_finalists': 1,
            'ranking': 'descending mean relative gain, lower effective new-model share, setting name, route',
            'behavior_tolerance_INR': 10., 'behavior_rmse_tolerance_INR': 10., 'prediction_rtol':1e-6,
            'candidate_same_layout_atol_INR': 10., 'v14_repeat_atol_INR': .01,
            'maximum_hours': 6, 'stage_seconds': __import__('core').STAGE_SECONDS,
            'sources': e.sources, 'inputs': e.inputs}
    path = e.out / 'experiment_plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan, 'Frozen experiment plan changed'
    save(path, plan)
    for scheme in DEVELOPMENT:
        baseline = e.reference(scheme)
        squared = (e.y - baseline) ** 2
        save(e.out / f'v14_error_audit_{scheme}.json', {
            'v14_rmse_INR': rmse(e.y, baseline),
            'tail_gt80M_rows': int((e.y > 8e7).sum()),
            'tail_gt80M_squared_error_share': float(squared[e.y > 8e7].sum() / squared.sum()),
            'top_five_squared_error_share': float(np.sort(squared)[-5:].sum() / squared.sum()),
            'label_errors_established': False})
    save(e.out / 'audit.json', {'status': 'PASS', 'sources': e.sources, 'inputs': e.inputs,
                              'all_original_targets': len(e.y), 'original_test_rows': len(e.test)})
    log('AUDIT PASS: three fixed candidate recipes, at most twenty comparisons, seven verified baseline assignments.')


def run_worker(e,name,arguments,allowance):
    import subprocess,sys,signal
    with (e.out/f'pilot_{name}.log').open('a') as log_handle:
        process=subprocess.Popen([sys.executable,'-u',*arguments],stdout=log_handle,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
        try:code=process.wait(timeout=max(1,allowance))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=-1
    return code


def pilot(e):
    config_path=ROOT/'resolved_models.json'
    if not config_path.exists():modeling.resolve()
    records={}
    controls=e.out/'control_pilot.json'
    if not controls.exists():
        assert run_worker(e,'CONTROLS',[str(ROOT/'control_worker.py')],min(1200,e.budget.remaining('pilot')-30))==0
    control_timings=json.loads(controls.read_text())
    mac=json.loads((ROOT/'mac_pilot.json').read_text());assert mac['status']=='PASS'
    for name in features.SPECS:
        path=e.out/'pilot_checkpoints'/f'{name}.json'
        if not path.exists():
            started=time.monotonic()
            code=run_worker(e,name,[str(ROOT/'pilot_worker.py'),name],min(2700 if name=='KS8-CLEAN' else 1500,e.budget.remaining('pilot')-60))
            if code!=0 or not path.exists():
                record={'status':'engineering_excluded','reason':'Bounded pilot process failed or timed out','accuracy_tested':False,'engineering_seconds':time.monotonic()-started}
                save(path,{'result':record,'result_sha256':digest(record)})
        data=json.loads(path.read_text());assert digest(data['result'])==data['result_sha256'];records[name]=data['result']
        save(e.out/'pilot_progress.json',records);log({'PILOT':name,'status':records[name]['status'],'reason':records[name].get('reason')})
    feasible=[n for n in features.SPECS if records[n]['status']=='PASS'];names=list(feasible);exclusions={}
    def forecasts(admitted):
        routes=features.routes(admitted)
        screen=1.3*sum(records[n]['fold_seconds']*10 for n in admitted)
        candidate_confirm=max([sum(records[n]['fold_seconds']*25 for n in route) for route in routes.values()] or [0])
        gpu_controls=15*(control_timings['BASE']['seconds']+control_timings['P35_8']['seconds'])
        mac_controls=15*mac['seconds']
        confirmation=1.3*(candidate_confirm+gpu_controls+mac_controls)
        export=1.3*(2*max([sum(records[n]['full_fit_predict_seconds'] for n in route) for route in routes.values()] or [0])+1800)
        return screen,confirmation,export,screen+confirmation<=e.budget.remaining('screen') and export<=e.budget.remaining('export')
    while names:
        screen_forecast,confirm_forecast,export_forecast,fits=forecasts(names)
        if fits:break
        dropped=next(n for n in ['I16','KS8-CLEAN','I8'] if n in names)
        names.remove(dropped);exclusions[dropped]='Complete workload plus thirty-percent contingency exceeds research pool or delivery reserve'
    screen_forecast,confirm_forecast,export_forecast,fits=forecasts(names)
    result={'status':'PASS' if names else 'NO_ADMITTED_SETTINGS','settings':records,'screen_settings':names,'routes':features.routes(names),
        'engineering_feasible_settings':feasible,'budget_exclusions':exclusions,'screen_forecast_seconds':screen_forecast,
        'confirmation_forecast_seconds':confirm_forecast,'full_data_and_fresh_notebook_forecast_seconds':export_forecast,
        'forecast_fits_budget':bool(names and fits),'selection_uses_validation_rmse':False,'sources':e.sources,'inputs':e.inputs,
        'model_contract_sha256':sha(ROOT/'resolved_models.json'),'control_timings':control_timings,'original_mac_control_timing':mac}
    save(e.out/'pilot.json',result);log({'PILOT_COMPLETE':True,'admitted':names,'forecasts_hours':[x/3600 for x in [screen_forecast,confirm_forecast,export_forecast]]})


def diagnostics(e, scheme, baseline, candidate, name):
    segments, folds = [], []
    cities = features.canonical_city(e.train['City'])
    unseen = np.zeros(len(e.y), bool)
    for _, (a, b) in enumerate(e.splits(scheme)):
        unseen[b] = ~cities.iloc[b].isin(set(cities.iloc[a].dropna())).to_numpy()
    for label, mask in [('all', np.ones(len(e.y), bool)), ('<=40M', e.y <= 4e7),
                        ('40-80M', (e.y > 4e7) & (e.y <= 8e7)), ('>80M', e.y > 8e7),
                        ('missing-city', cities.isna().to_numpy()), ('unseen-city', unseen)]:
        if not mask.any():
            continue
        old, new = rmse(e.y[mask], baseline[mask]), rmse(e.y[mask], candidate[mask])
        segments.append({'name': name, 'scheme': str(scheme), 'segment': label, 'rows': int(mask.sum()),
                         'v14_rmse': old, 'candidate_rmse': new, 'gain': old - new})
    for fold, (_, b) in enumerate(e.splits(scheme)):
        old, new = rmse(e.y[b], baseline[b]), rmse(e.y[b], candidate[b])
        folds.append({'name': name, 'scheme': str(scheme), 'fold': fold, 'rows': len(b),
                      'v14_rmse': old, 'candidate_rmse': new, 'gain': old - new})
    delta = (e.y - baseline) ** 2 - (e.y - candidate) ** 2
    top = np.argsort(-delta)[:10]
    concentration = {'name': name, 'scheme': str(scheme), 'net_squared_error_reduction': float(delta.sum()),
                     'largest_positive_ten_rows': e.train.iloc[top]['Registration Number'].tolist(),
                     'top_ten_positive_reduction': float(np.maximum(delta[top], 0).sum()),
                     'tail_gt80M_reduction': float(delta[e.y > 8e7].sum())}
    largest_baseline_errors = np.argsort((e.y-baseline)**2)[-5:]
    mask = np.ones(len(e.y), bool)
    mask[largest_baseline_errors] = False
    concentration.update(excluded_baseline_top_five_ids=e.train.iloc[largest_baseline_errors]['Registration Number'].tolist(),
                         gain_excluding_baseline_top_five_INR=rmse(e.y[mask], baseline[mask])-rmse(e.y[mask], candidate[mask]),
                         exclusion_is_selection_criterion=False)
    return segments, folds, concentration


def selection(rows):
    eligible=[r for r in rows if r['gain_42']>0 and r['gain_71']>0 and r['mean_relative_gain']>=.002]
    return sorted(eligible,key=lambda r:(-r['mean_relative_gain'],r['weight'],r['components'],r['route']))[:1]


def screen(e):
    pilot=json.loads((e.out/'pilot.json').read_text());assert pilot['sources']==e.sources and pilot['inputs']==e.inputs
    assert pilot['model_contract_sha256']==sha(ROOT/'resolved_models.json')
    rows,segments,folds,concentrations,components=[],[],[],[],[]
    refs={s:e.reference_parts(s) for s in DEVELOPMENT};predictions={}
    if any(n.startswith('FT') for n in pilot['screen_settings']):
        for scheme in DEVELOPMENT:
            q,_,meta=e.component('UNADAPTED',scheme)
            save(e.out/f'matched_unadapted_{scheme}.json',{'rmse':rmse(e.y,q),'fit_metadata':meta,'diagnostic_only':True})
    for name in pilot['screen_settings']:
        predictions[name]={}
        for scheme in DEVELOPMENT:
            q,_,meta=e.component(name,scheme);predictions[name][scheme]=q
            components.append({'name':name,'scheme':scheme,'component_rmse':rmse(e.y,q),'v14_rmse':rmse(e.y,refs[scheme]['v14']),
                'error_correlation':float(np.corrcoef(e.y-q,e.y-refs[scheme]['v14'])[0,1]),'error_correlation_le80M':float(np.corrcoef((e.y-q)[e.y<=8e7],(e.y-refs[scheme]['v14'])[e.y<=8e7])[0,1])})
            save(e.out/f'fit_metadata_{name}_{scheme}.json',meta)
    routes=pilot['routes'];assert all(set(v)<=set(predictions) for v in routes.values())
    for route,weights in routes.items():
        for weight in features.WEIGHTS:
            row={'name':route,'route':route,'weight':weight,'effective_new_share':weight,'components':len(weights)}
            for scheme in DEVELOPMENT:
                parts=refs[scheme];q=sum(w*predictions[n][scheme] for n,w in weights.items())
                candidate=modeling.combine(parts['v9'],q,weight,parts['p35']);old,new=rmse(e.y,parts['v14']),rmse(e.y,candidate)
                row.update({f'rmse_{scheme}':new,f'gain_{scheme}':old-new,f'relative_gain_{scheme}':(old-new)/old})
                seg,fld,con=diagnostics(e,scheme,parts['v14'],candidate,f'{route}@{weight}');segments+=seg;folds+=fld;concentrations.append(con)
            row['mean_relative_gain']=float(np.mean([row[f'relative_gain_{s}'] for s in DEVELOPMENT]));rows.append(row)
    columns=['name','route','weight','effective_new_share','components','rmse_42','gain_42','relative_gain_42','rmse_71','gain_71','relative_gain_71','mean_relative_gain']
    pd.DataFrame(rows,columns=columns).to_csv(e.out/'screening.csv',index=False)
    for filename,data in [('development_segments.csv',segments),('development_folds.csv',folds),('component_diagnostics.csv',components)]:
        pd.DataFrame(data).to_csv(e.out/filename,index=False)
    save(e.out/'development_concentration.json',concentrations)
    assert len(rows)==len(routes)*len(features.WEIGHTS)<=20
    lock={'complete':True,'choices':[dict(r,component_weights=routes[r['route']]) for r in selection(rows)],'settings':len(predictions),'configurations':len(rows),'sources':e.sources,'inputs':e.inputs,
        'model_contract_sha256':sha(ROOT/'resolved_models.json'),'experiment_plan_sha256':sha(e.out/'experiment_plan.json'),'frozen_before_confirmation':True}
    path=e.out/'candidate_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock,'Frozen finalist changed'
    save(path,lock);log({'SCREEN_COMPLETE':True,'choices':lock['choices']})


def uncertainty(y, baselines, predictions):
    rng = np.random.default_rng(8319)
    gains = np.empty(5000)
    original = np.stack([(y - baselines[s]) ** 2 for s in CONFIRM[:3]])
    candidate = np.stack([(y - predictions[s]) ** 2 for s in CONFIRM[:3]])
    for i in range(5000):
        rows = rng.integers(0, len(y), len(y))
        gains[i] = np.mean(np.sqrt(original[:, rows].mean(axis=1)) - np.sqrt(candidate[:, rows].mean(axis=1)))
    return {'resamples': 5000, 'seed': 8319, 'ci95_INR': np.quantile(gains, [.025, .975]).tolist(),
            'interpretation': 'Conditional paired-row robustness interval; excludes selection, retraining and hidden-test shift.'}


def qualifies(metrics, interval):
    mean = float(np.mean([metrics[str(s)]['relative_gain'] for s in CONFIRM[:3]]))
    return bool(all(metrics[str(s)]['gain'] > 0 for s in CONFIRM[:3]) and mean >= .002
                and interval['ci95_INR'][0] > 0
                and all(metrics[s]['relative_gain'] >= -.01 for s in ['strat:27421', 'city']))


def confirm(e):
    lock = json.loads((e.out / 'candidate_lock.json').read_text())
    assert lock['sources'] == e.sources and lock['inputs'] == e.inputs
    assert lock['model_contract_sha256'] == sha(ROOT / 'resolved_models.json')
    if not lock['choices']:
        save(e.out / 'decision.json', {'complete': True, 'qualified': False, 'choices': [], 'finalists': [],
                                      'decision': 'Retain V14: no development candidate qualified.', 'portal_uploads': 0})
        return
    choice = lock['choices'][0]
    metrics, baselines, predictions = {}, {}, {}
    segments, folds, concentrations = [], [], []
    for scheme in CONFIRM:
        parts = e.reference_parts(scheme)
        base = parts['v14']
        component=sum(weight*e.component(name,scheme)[0] for name,weight in choice['component_weights'].items())
        if any(name.startswith('FT') for name in choice['component_weights']):e.component('UNADAPTED',scheme)
        candidate = modeling.combine(parts['v9'], component, choice['weight'], parts['p35'])
        old, new = rmse(e.y, base), rmse(e.y, candidate)
        metrics[str(scheme)] = {'v14_rmse': old, 'candidate_rmse': new, 'gain': old - new,
                                'relative_gain': (old - new) / old}
        baselines[scheme], predictions[scheme] = base, candidate
        np.savez_compressed(e.out / f"confirmation_{str(scheme).replace(':', '_')}.npz", baseline=base, candidate=candidate)
        seg, fld, con = diagnostics(e, scheme, base, candidate, choice['name'])
        segments.extend(seg)
        folds.extend(fld)
        concentrations.append(con)
        log({'CONFIRMATION': str(scheme), **metrics[str(scheme)]})
        save(e.out / 'confirmation_progress.json', {'metrics': metrics, 'lock_sha256': sha(e.out / 'candidate_lock.json')})
    interval = uncertainty(e.y, baselines, predictions)
    qualified = qualifies(metrics, interval)
    finalist = {'choice': choice, 'qualified': qualified, 'metrics': metrics, 'uncertainty': interval}
    pd.DataFrame([{'scheme': s, **m} for s, m in metrics.items()]).to_csv(e.out / 'confirmation.csv', index=False)
    pd.DataFrame(segments).to_csv(e.out / 'confirmation_segments.csv', index=False)
    pd.DataFrame(folds).to_csv(e.out / 'confirmation_folds.csv', index=False)
    save(e.out / 'confirmation_concentration.json', concentrations)
    save(e.out / 'decision.json', {'complete': True, 'qualified': qualified,
                                  'choices': [choice] if qualified else [], 'finalists': [finalist],
                                  'lock_sha256': sha(e.out / 'candidate_lock.json'),
                                  'decision': 'One qualified candidate for manual portal trial.' if qualified else 'Retain V14: frozen finalist failed confirmation.',
                                  'portal_uploads': 0})


def export(e):
    import delivery
    decision = json.loads((e.out / 'decision.json').read_text())
    target = ROOT / 'delivery'
    target.mkdir(exist_ok=True)
    if decision['qualified']:
        choice = decision['choices'][0]
        with (ROOT / 'reference/verified_v7.pkl').open('rb') as handle:
            original = pickle.load(handle)
        import baseline
        b = baseline.predict(original, e.test)
        model = modeling.fit(e.train, e.y, 'BASE')
        try:
            t = modeling.predict(model, e.test)
        finally:
            modeling.release(model)
        v9 = .5 * b + .5 * t
        expected_v9 = pd.read_csv(ROOT / 'reference/scored_v9.csv')['Annual Turnover'].to_numpy(float)
        difference = float(np.max(np.abs(v9 - expected_v9)))
        assert difference < .01, f'V9 reproduction differs INR {difference}'
        model = modeling.fit(e.train, e.y, 'P35_8')
        try:
            p35 = modeling.predict(model, e.test)
        finally:
            modeling.release(model)
        v14 = .25 * v9 + .75 * p35
        expected_v14 = pd.read_csv(ROOT / 'reference/scored_v14.csv')['Annual Turnover'].to_numpy(float)
        v14_difference = float(np.max(np.abs(v14 - expected_v14)))
        assert v14_difference < .01, f'V14 reproduction differs INR {v14_difference}'
        component=np.zeros(len(e.test));fit_metadata={}
        for name,weight in choice['component_weights'].items():
            model=modeling.fit(e.train,e.y,name)
            try:
                component+=weight*modeling.predict(model,e.test);fit_metadata[name]=model['target_metadata']
            finally:modeling.release(model)
        prediction = modeling.combine(v9, component, choice['weight'], p35)
        checks = verify.csv_checks(e, prediction)
        frame = pd.DataFrame({'Registration Number': e.test['Registration Number'], 'Annual Turnover': prediction})
        validate_csv(frame, e.test)
        frame.to_csv(target / (STEM + '.csv'), index=False)
        save(target / 'initial_export.json', {'v9_reproduction_difference_INR': difference,
                                            'v14_reproduction_difference_INR': v14_difference,
                                            'target_metadata': fit_metadata, 'csv_checks': checks})
        delivery.build_and_execute(e, decision, target)
    else:
        import shutil
        for suffix in ['.csv', '.ipynb']:
            shutil.copy2(ROOT / 'reference' / ('scored_v14' + suffix), target / ('KEEP_V14_' + STEM + suffix))
        save(target / 'fallback_verification.json', {'status': 'PASS', 'v14_files_byte_identical': True,
                                                     'new_v20_csv_exported': False, 'fresh_v20_notebook_executed': False})
    delivery.package(e, decision, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'pilot', 'screen', 'confirm', 'export'])
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()
    e = Engine(args.data_dir, args.output_dir)
    if args.stage != 'audit':
        previous = json.loads((e.out / 'audit.json').read_text())
        assert previous['sources'] == e.sources and previous['inputs'] == e.inputs
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(15):
            save(e.out / 'runtime_heartbeat.json', {'stage': args.stage, 'epoch': time.time(), 'status': 'running'})
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    save(e.out / 'runtime_heartbeat.json', {'stage': args.stage, 'epoch': time.time(), 'status': 'running'})
    try:
        with e.budget.charge(args.stage):
            os.environ['MLA_V20_TRAINING_DEADLINE'] = str(time.time() + e.budget.remaining(args.stage))
            globals()[args.stage](e)
        save(e.out / 'stage_status.json', {'stage': args.stage, 'status': 'complete'})
    except Exception as error:
        save(e.out / 'stage_status.json', {'stage': args.stage, 'status': 'failed',
                                         'error_type': type(error).__name__, 'reason': str(error)[:1000]})
        raise
    finally:
        stop.set()
        worker.join(timeout=2)
        save(e.out / 'runtime_heartbeat.json', {'stage': args.stage, 'epoch': time.time(), 'status': 'finished'})


if __name__ == '__main__':
    main()

