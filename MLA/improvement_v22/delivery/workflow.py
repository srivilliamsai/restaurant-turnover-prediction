"""One fixed survey-pair projection, three comparisons and frozen confirmation."""
import argparse
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
            'formula': '.25*V9+.75*((1-w)*P35_original+w*M)', 'control': 'V14',
            'minimum_mean_relative_gain': .002,
            'maximum_stress_deterioration': .01, 'maximum_finalists': 1,
            'ranking': 'descending mean relative gain, lower effective new-model share, setting name, route',
            'behavior_tolerance_INR': 10., 'behavior_rmse_tolerance_INR': 1.,
            'same_layout_repeat_tolerance_INR': .01,
            'maximum_hours': 4, 'stage_seconds': __import__('core').STAGE_SECONDS,
            'sources': e.sources, 'inputs': e.inputs}
    path = e.out / 'experiment_plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan, 'Frozen experiment plan changed'
    save(path, plan)
    save(e.out / 'audit.json', {'status': 'PASS', 'sources': e.sources, 'inputs': e.inputs,
                              'all_original_targets': len(e.y), 'original_test_rows': len(e.test)})
    log('AUDIT PASS: one fixed component, three comparisons, seven verified baseline assignments.')


def pilot(e):
    config_path = ROOT / 'resolved_models.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else modeling.resolve()
    assert config['runtime'] == modeling.runtime()
    a, b = e.splits(42)[0]
    control_path = e.out / 'control_pilot.json'
    if control_path.exists():
        control_record = json.loads(control_path.read_text())
        assert control_record['sources'] == e.sources
        assert control_record['model_contract_sha256'] == sha(config_path)
    else:
        control_record = {'sources': e.sources, 'model_contract_sha256': sha(config_path), 'controls': {}}
        reference = e.reference_parts(42)
        for control_name, reference_key in [('BASE', 't9'), ('P35_8', 'p35')]:
            e.budget.require(60)
            started = time.monotonic()
            control = modeling.fit(e.train.iloc[a], e.y[a], control_name)
            try:
                predicted = modeling.predict(control, e.train.iloc[b])
                difference = float(np.max(np.abs(predicted - reference[reference_key][b])))
                assert difference < .01, f'{control_name} control reproduction differs INR {difference}'
            finally:
                modeling.release(control)
            control_record['controls'][control_name] = {'seconds': time.monotonic() - started,
                                                       'maximum_difference_INR': difference}
        control_record['status'] = 'PASS'
        save(control_path, control_record)
        log({'CONTROL_PILOT': control_record})
    records = {}
    for name in features.SPECS:
        e.budget.require(30)
        identity = {'name': name, 'sources': e.sources, 'inputs': e.inputs,
                    'runtime_contract': config, 'train_indices': digest(a.tolist()),
                    'valid_indices': digest(b.tolist()),
                    'training_targets': hashlib.sha256(e.y[a].tobytes()).hexdigest()}
        path = e.out / 'pilot_checkpoints' / (name + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            assert record['context'] == identity and record['result_sha256'] == digest(record['result'])
            records[name] = record['result']
            continue
        fitted = repeated = None
        started = time.monotonic()
        try:
            assert name in config['models'], config['engineering_failures'].get(name, 'Model unavailable')
            fitted = modeling.fit(e.train.iloc[a], e.y[a], name)
            full = modeling.predict_outputs(fitted, e.train.iloc[b])
            fold_seconds = time.monotonic() - started
            target_metadata = fitted['target_metadata']
            checks = verify.inference_checks(lambda raw: modeling.predict(fitted, raw), e.train.iloc[b], e.y[b])
            changed_surveys = e.train.iloc[b].copy()
            changed_surveys['Service'] = 1e20
            changed_surveys['Comfortablility'] = -1e20
            removed_difference = float(np.max(np.abs(modeling.predict(fitted, changed_surveys) - full['prediction'])))
            assert removed_difference == 0., 'Removed survey values influenced the candidate'
            checks['removed_feature_mutation_difference_INR'] = removed_difference
            import torch
            peak_memory = torch.cuda.max_memory_allocated()
            modeling.release(fitted)
            fitted = None
            changed = e.train.iloc[b].copy()
            changed['Annual Turnover'] = 1e30
            changed['Registration Number'] = -1
            repeated = modeling.fit(e.train.iloc[a], e.y[a], name)
            other = modeling.predict_outputs(repeated, changed)
            difference = max(float(np.max(np.abs(full[k] - other[k]))) for k in full)
            assert difference < .01, f'Fresh repeated fit differs INR {difference}'
            modeling.release(repeated)
            repeated = None
            full_started = time.monotonic()
            fitted = modeling.fit(e.train, e.y, name)
            full_predictions = modeling.predict(fitted, e.test)
            assert full_predictions.shape == (500,) and np.isfinite(full_predictions).all()
            full_data_seconds = time.monotonic() - full_started
            import pickle
            serialized = pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL)
            modeling.release(fitted)
            fitted = None
            repeated = pickle.loads(serialized)
            del serialized
            restored = modeling.predict(repeated, e.test)
            reload_difference = float(np.max(np.abs(full_predictions - restored)))
            assert reload_difference < .01, f'Reload differs INR {reload_difference}'
            result = {'status': 'PASS', 'fold_seconds': fold_seconds, 'behavior': checks,
                      'fresh_repeat_difference_INR': difference, 'target_metadata': target_metadata,
                      'full_data_seconds': full_data_seconds, 'full_data_training_rows': len(e.train),
                      'full_data_query_rows': len(e.test), 'reload_difference_INR': reload_difference,
                      'peak_gpu_allocated_bytes': max(peak_memory, torch.cuda.max_memory_allocated()),
                      'training_rows': len(a), 'validation_rows': len(b),
                      'engineering_seconds': time.monotonic() - started}
        except Exception as error:
            import traceback
            result = {'status': 'unavailable', 'error_type': type(error).__name__, 'reason': str(error)[:800],
                      'traceback': traceback.format_exc(),
                      'engineering_seconds': time.monotonic() - started}
        finally:
            modeling.release(fitted)
            modeling.release(repeated)
        records[name] = result
        save(path, {'context': identity, 'result': result, 'result_sha256': digest(result)})
        save(e.out / 'pilot_progress.json', records)
        log({'PILOT': name, **result})
    names = [name for name in features.SPECS if records[name]['status'] == 'PASS']
    engineering_feasible_names = names.copy()
    # Feasibility exclusions are decided solely from measured engineering costs.
    exclusions = {}
    remaining = max(0., min(e.budget.authorization['training_stop_epoch'] - time.time(),
                            e.budget.authorization['hard_stop_epoch'] - time.time() - 5400))
    screen_forecast = 1.3 * sum(records[n]['fold_seconds'] * 10 for n in names)
    confirmation_forecast = 1.3 * max([records[n]['fold_seconds'] * 25 for n in names] or [0])
    delivery_forecast = 1.3 * (6 * max([records[n]['full_data_seconds'] for n in names] or [0]) + 300)
    fits = (screen_forecast <= e.budget.remaining('screen') and
            confirmation_forecast <= e.budget.remaining('confirm') and
            delivery_forecast <= e.budget.remaining('export') and
            screen_forecast + confirmation_forecast <= remaining and
            screen_forecast + confirmation_forecast + delivery_forecast <=
            e.budget.authorization['hard_stop_epoch'] - time.time())
    result = {'status': 'PASS' if names and fits else 'NO_FEASIBLE_SETTINGS' if not names else 'BUDGET_FORECAST_EXCEEDED',
              'settings': records, 'screen_settings': names if fits else [],
              'engineering_feasible_settings': engineering_feasible_names, 'budget_exclusions': exclusions,
              'screen_forecast_seconds': screen_forecast, 'confirmation_forecast_seconds': confirmation_forecast,
              'delivery_forecast_seconds': delivery_forecast,
              'forecast_fits_budget': bool(fits), 'selection_uses_validation_rmse': False,
              'sources': e.sources, 'inputs': e.inputs, 'model_contract_sha256': sha(config_path)}
    save(e.out / 'pilot.json', result)
    log({'PILOT_COMPLETE': result['status'], 'screen_settings': result['screen_settings']})


def diagnostics(e, scheme, baseline, candidate, name):
    segments, folds = [], []
    for label, mask in [('all', np.ones(len(e.y), bool)), ('<=40M', e.y <= 4e7),
                        ('40-80M', (e.y > 4e7) & (e.y <= 8e7)), ('>80M', e.y > 8e7)]:
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
    concentration['survey_bin_prediction_changes'] = []
    for column in features.DROP:
        values = e.train[column].to_numpy(float)
        for label, mask in [('0-3', values <= 3), ('4-6', (values >= 4) & (values <= 6)),
                            ('7-10', values >= 7)]:
            if mask.any():
                change = candidate[mask] - baseline[mask]
                concentration['survey_bin_prediction_changes'].append({
                    'column': column, 'bin': label, 'rows': int(mask.sum()),
                    'mean_prediction_change_INR': float(change.mean()),
                    'mean_absolute_prediction_change_INR': float(np.abs(change).mean()),
                    'used_for_selection': False})
    return segments, folds, concentration


def selection(rows):
    eligible = [r for r in rows if r['gain_42'] > 0 and r['gain_71'] > 0 and r['mean_relative_gain'] >= .002]
    return sorted(eligible, key=lambda r: (-r['mean_relative_gain'], r['effective_new_share'], r['name'], r['route']))[:1]


def screen(e):
    pilot = json.loads((e.out / 'pilot.json').read_text())
    assert pilot['sources'] == e.sources and pilot['inputs'] == e.inputs
    assert pilot['model_contract_sha256'] == sha(ROOT / 'resolved_models.json')
    rows, segments, folds, concentrations, components = [], [], [], [], []
    refs = {s: e.reference_parts(s) for s in DEVELOPMENT}
    for name in pilot['screen_settings']:
        predictions = {}
        for scheme in DEVELOPMENT:
            component, _, metadata = e.component(name, scheme)
            predictions[scheme] = component
            components.append({'name': name, 'scheme': scheme,
                               'component_rmse': rmse(e.y, component),
                               'v14_rmse': rmse(e.y, refs[scheme]['v14']),
                               'error_correlation': float(np.corrcoef(e.y-component, e.y-refs[scheme]['v14'])[0, 1])})
            save(e.out / f'fit_metadata_{name}_{scheme}.json', metadata)
        for route in features.ROUTES:
            for weight in features.WEIGHTS:
                row = {'name': name, 'route': route, 'weight': weight,
                       'effective_new_share': .75 * weight}
                for scheme in DEVELOPMENT:
                    parts = refs[scheme]
                    base = parts['v14']
                    candidate = modeling.combine(parts['v9'], predictions[scheme], weight, parts['p35'])
                    old, new = rmse(e.y, base), rmse(e.y, candidate)
                    row.update({f'rmse_{scheme}': new, f'gain_{scheme}': old-new,
                                f'relative_gain_{scheme}': (old-new)/old})
                    seg, fld, con = diagnostics(e, scheme, base, candidate, f'{name}_{route}@{weight}')
                    segments.extend(seg)
                    folds.extend(fld)
                    concentrations.append(con)
                row['mean_relative_gain'] = float(np.mean([row[f'relative_gain_{s}'] for s in DEVELOPMENT]))
                rows.append(row)
        for filename, data in [('screening.csv', rows), ('development_segments.csv', segments),
                               ('development_folds.csv', folds), ('component_diagnostics.csv', components)]:
            pd.DataFrame(data).to_csv(e.out / filename, index=False)
        save(e.out / 'development_concentration.json', concentrations)
        log({'SCREEN_SETTING_COMPLETE': name, 'best_mean_relative_gain': max(r['mean_relative_gain'] for r in rows if r['name'] == name)})
    if not rows:
        pd.DataFrame(columns=['name','route','weight','effective_new_share','rmse_42','gain_42','relative_gain_42','rmse_71','gain_71','relative_gain_71','mean_relative_gain']).to_csv(e.out / 'screening.csv', index=False)
    assert len(rows) == len(pilot['screen_settings']) * len(features.WEIGHTS)
    lock = {'complete': True, 'choices': selection(rows), 'settings': len(pilot['screen_settings']),
            'configurations': len(rows), 'sources': e.sources, 'inputs': e.inputs,
            'model_contract_sha256': sha(ROOT / 'resolved_models.json'),
            'experiment_plan_sha256': sha(e.out / 'experiment_plan.json'), 'frozen_before_confirmation': True}
    path = e.out / 'candidate_lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == lock, 'Frozen finalist changed'
    save(path, lock)
    log({'SCREEN_COMPLETE': True, 'choices': lock['choices']})


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
        component, naive, targets = e.component(choice['name'], scheme)
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
        model = modeling.fit(e.train, e.y, choice['name'])
        try:
            component = modeling.predict(model, e.test)
            fit_metadata = model['target_metadata']
        finally:
            modeling.release(model)
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
                                                     'new_v22_csv_exported': False, 'fresh_v22_notebook_executed': False})
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

