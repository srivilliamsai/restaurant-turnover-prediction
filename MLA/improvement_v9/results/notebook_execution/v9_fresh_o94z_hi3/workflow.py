"""The fixed V9 audit, screening, refinement, confirmation and export stages."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import modeling as models
import baseline
import pilot as pilot_module
from engine import (ROOT, STEM, CONFIRM, RANDOM_CONFIRM, Engine, BudgetExceeded,
                    sha, digest, save, log, safe, rmse, check_behavior, validate_csv)


def audit(e):
    m = e.manifest
    for n, expected in m['frozen_files'].items():
        assert sha(ROOT / n) == expected
    for n, expected in m['preserved_v7'].items():
        assert sha(ROOT / 'scored_v7_backup' / n) == expected
    restored = (ROOT / 'legacy_models.py').read_text().replace('import legacy_reference as ref', 'import reference as ref')
    import hashlib
    assert hashlib.sha256(restored.encode()).hexdigest() == m['original_v7_sources']['models.py']
    assert sha(ROOT / 'legacy_reference.py') == m['original_v7_sources']['reference.py']
    changed = e.train.copy()
    changed['Annual Turnover'] = 1e30
    changed['Registration Number'] = -9
    pd.testing.assert_frame_equal(models.features(changed), models.features(e.train))
    assert len(e.y) == 3493 and np.isclose(rmse([0, 3], [0, 0]), np.sqrt(4.5))
    scored = pd.read_csv(ROOT / 'scored_v7_backup' / f'{STEM}.csv')
    validate_csv(scored, e.test)
    for scheme in [42, 71, 'strat:27421', 'city']:
        p = e.baseline(scheme)
        assert p.shape == e.y.shape and np.isfinite(p).all()
    record = {'inputs': e.inputs, 'sources': e.sources, 'runtime': e.runtime,
        'python': __import__('platform').python_version(), 'rows': len(e.y), 'test_rows': len(e.test),
        'labels_changed': 0, 'tail_above_80M': int((e.y > 80e6).sum()),
        'scored_v7_preserved': m['preserved_v7'], 'scored_v7_portal_rmse': baseline.PORTAL_RMSE,
        'utc': datetime.now(timezone.utc).isoformat(), 'external_code_or_predictions_used': False}
    save(e.out / 'audit.json', record)
    plan = {'specs': models.SPECS, 'weights': models.WEIGHTS, 'development': [42, 71],
        'confirmation': CONFIRM, 'max_base_settings': 8, 'max_total_settings': 10,
        'max_development_configurations': 40, 'refinement_modes': ['ensemble16'],
        'resolved_defaults': models.resolved_defaults(),
        'shortlist': 'Best eligible TabPFN and EBM group; gain positive on both development assignments',
        'qualification': 'Gain positive on all three random confirmations; city and stratified deterioration <=1%',
        'budget_hours': e.cache.hours, 'no_search_expansion_after_confirmation': True}
    path = e.out / 'experiment_plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    save(path, plan)
    log({'audit': 'PASS', 'settings': len(models.SPECS), 'v7_portal_rmse': baseline.PORTAL_RMSE})


def pilot(e):
    pilot_module.run(e)


def comparison(e, name, mode, baselines):
    p = {s: e.component(name, mode, s) for s in [42, 71]}
    rows = []
    for w in models.WEIGHTS:
        row = {'name': name, 'mode': mode, 'group': models.SPECS[name]['group'],
               'family': models.SPECS[name]['family'], 'weight': w}
        for s in [42, 71]:
            row[f'rmse_{s}'] = rmse(e.y, (1 - w) * baselines[s] + w * p[s])
            row[f'gain_{s}'] = rmse(e.y, baselines[s]) - row[f'rmse_{s}']
        row['mean_rmse'] = float(np.mean([row['rmse_42'], row['rmse_71']]))
        rows.append(row)
    return rows


def ranked(table):
    return table.sort_values(['mean_rmse', 'weight', 'name', 'mode'])


def eligible(table):
    return ranked(table[(table.gain_42 > 0) & (table.gain_71 > 0)])


def screen(e):
    b = {s: e.baseline(s) for s in [42, 71]}
    rows = []
    available = e.available_specs()
    assert available, 'No family passed the feasibility pilot.'
    for name in available:
        rows.extend(comparison(e, name, 'plain', b))
        ranked(pd.DataFrame(rows)).to_csv(e.out / 'screening.csv', index=False)
        log({'screen_complete': name, 'settings_finished': len(rows) // 4, 'of': len(available)})
    table = ranked(pd.DataFrame(rows))
    short = eligible(table).groupby('group', sort=False).head(1).to_dict(orient='records')
    save(e.out / 'screening_decision.json', {'complete': True, 'settings': len(available), 'planned_settings': 8,
        'skipped_settings': sorted(set(models.SPECS) - set(available)), 'configurations': len(rows), 'shortlist': short})
    log({'screening_shortlist': short})


def refine(e):
    selected = json.loads((e.out / 'screening_decision.json').read_text())
    assert selected['complete'] and selected['settings'] == len(e.available_specs())
    b = {s: e.baseline(s) for s in [42, 71]}
    rows = pd.read_csv(e.out / 'screening.csv').to_dict(orient='records')
    for choice in selected['shortlist']:
        for mode in ['ensemble16']:
            rows.extend(comparison(e, choice['name'], mode, b))
            ranked(pd.DataFrame(rows)).to_csv(e.out / 'development.csv', index=False)
    table = ranked(pd.DataFrame(rows))
    table.to_csv(e.out / 'development.csv', index=False)
    permitted = {r['name'] for r in selected['shortlist']}
    short = eligible(table[table['name'].isin(permitted)]).groupby('group', sort=False).head(1).to_dict(orient='records')
    lock = {'shortlist': short, 'inputs': e.inputs, 'sources': e.sources,
        'plan_sha256': sha(e.out / 'experiment_plan.json'), 'development_configurations': len(table),
        'frozen_before_confirmation': True}
    path = e.out / 'candidate_lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == lock, 'Frozen shortlist changed.'
    save(path, lock)
    log({'FROZEN': short, 'lock_sha256': sha(path), 'development_configurations': len(table)})


def uncertainty(y, bases, predictions):
    rng = np.random.default_rng(8319)
    gains = []
    for _ in range(50):
        index = rng.integers(0, len(y), (100, len(y)))
        one = []
        for s in RANDOM_CONFIRM:
            be = (y - bases[s]) ** 2
            pe = (y - predictions[s]) ** 2
            one.append(np.sqrt(be[index].mean(axis=1)) - np.sqrt(pe[index].mean(axis=1)))
        gains.extend(np.mean(one, axis=0).tolist())
    return {'repetitions': 5000, 'seed': 8319, 'ci95': np.quantile(gains, [.025, .975]).tolist(),
            'positive_fraction': float(np.mean(np.asarray(gains) > 0)),
            'interpretation': 'Conditional joint restaurant-row resampling across three assignments; excludes search, retraining and distribution uncertainty.'}


def confirm(e):
    lock = json.loads((e.out / 'candidate_lock.json').read_text())
    assert lock['sources'] == e.sources and lock['inputs'] == e.inputs
    rows, segments, folds, decisions = [], [], [], []
    bases = {}
    for choice in lock['shortlist']:
        predictions, metrics = {}, {}
        for scheme in CONFIRM:
            if scheme not in bases:
                bases[scheme] = e.baseline(scheme)
            b = bases[scheme]
            c = e.component(choice['name'], choice['mode'], scheme)
            p = (1 - choice['weight']) * b + choice['weight'] * c
            predictions[scheme] = p
            old, new = rmse(e.y, b), rmse(e.y, p)
            metrics[str(scheme)] = {'v7_rmse': old, 'candidate_rmse': new, 'gain': old - new, 'relative_change': new / old - 1}
            rows.append({**{k: choice[k] for k in ['name', 'mode', 'group', 'weight']}, 'scheme': str(scheme), **metrics[str(scheme)]})
            for label, mask in [('<=40M', e.y <= 40e6), ('40-80M', (e.y > 40e6) & (e.y <= 80e6)), ('>80M', e.y > 80e6)]:
                total_reduction = float(((e.y - b) ** 2 - (e.y - p) ** 2).sum())
                reduction = float(((e.y[mask] - b[mask]) ** 2 - (e.y[mask] - p[mask]) ** 2).sum())
                segments.append({'name': choice['name'], 'mode': choice['mode'], 'scheme': str(scheme), 'segment': label,
                    'rows': int(mask.sum()), 'v7_rmse': rmse(e.y[mask], b[mask]), 'candidate_rmse': rmse(e.y[mask], p[mask]),
                    'candidate_sse_share': float(((e.y[mask] - p[mask]) ** 2).sum() / ((e.y - p) ** 2).sum()),
                    'sse_reduction': reduction, 'fraction_of_total_reduction': reduction / total_reduction if total_reduction else None})
            for f, (_, indices) in enumerate(e.splits(scheme)):
                folds.append({'name': choice['name'], 'mode': choice['mode'], 'scheme': str(scheme), 'fold': f,
                    'rows': len(indices), 'tail_rows': int((e.y[indices] > 80e6).sum()),
                    'v7_rmse': rmse(e.y[indices], b[indices]), 'candidate_rmse': rmse(e.y[indices], p[indices])})
            pd.DataFrame(rows).to_csv(e.out / 'confirmation.csv', index=False)
            pd.DataFrame(segments).to_csv(e.out / 'segments.csv', index=False)
            pd.DataFrame(folds).to_csv(e.out / 'folds.csv', index=False)
            log({'confirmation': choice['name'], 'scheme': str(scheme), **metrics[str(scheme)]})
        qualified = all(metrics[str(s)]['gain'] > 0 for s in RANDOM_CONFIRM) and all(
            metrics[str(s)]['relative_change'] <= .01 for s in ['strat:27421', 'city'])
        decisions.append({**choice, 'metrics': metrics, 'qualified': bool(qualified),
            'mean_random_gain': float(np.mean([metrics[str(s)]['gain'] for s in RANDOM_CONFIRM])),
            'uncertainty': uncertainty(e.y, bases, predictions)})
    trials = sorted([d for d in decisions if d['qualified']], key=lambda d: (-d['mean_random_gain'], d['weight'], d['name']))
    save(e.out / 'decision.json', {'candidates': decisions, 'trials': trials,
        'lock_sha256': sha(e.out / 'candidate_lock.json'), 'baseline_portal_rmse': baseline.PORTAL_RMSE,
        'decision': 'qualified model-generated portal trials' if trials else 'retain scored V7; no V9 candidate passed all gates',
        'portal_uploads_by_agent': 0})
    log({'DECISION': 'qualified portal trials' if trials else 'retain V7', 'qualified': len(trials)})


def _export(e):
    decision = json.loads((e.out / 'decision.json').read_text())
    # Always perform separate fresh V7 reproduction, even when no V9 model qualifies.
    log('Fresh scored-V7 full reproduction begins.')
    started = __import__('time').monotonic()
    b, fitted_base = baseline.fit_full(e.train, e.test, e.y, e.contract, log)
    expected = pd.read_csv(ROOT / 'scored_v7_backup' / f'{STEM}.csv')['Annual Turnover'].to_numpy(float)
    difference = float(np.max(np.abs(b - expected)))
    assert difference < .01
    base_checks = check_behavior(lambda raw: baseline.predict(fitted_base, raw), e.test, b)
    save(e.out / 'v7_full_reproduction.json', {'maximum_difference_INR': difference,
        'fresh_full_fit': True, 'seconds': __import__('time').monotonic() - started, 'checks': base_checks})
    checks = []
    for i, choice in enumerate(decision['trials'], 1):
        fitted = models.fit_model(e.train, e.y, models.SPECS[choice['name']], choice['mode'], log)
        p = (1 - choice['weight']) * b + choice['weight'] * models.predict_model(fitted, e.test)
        predict = lambda raw: ((1 - choice['weight']) * baseline.predict(fitted_base, raw)
                              + choice['weight'] * models.predict_model(fitted, raw))
        check = check_behavior(predict, e.test, p)
        directory = e.out.parent / 'candidates' / f'trial_{i}'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{STEM}.csv'
        pd.DataFrame({'Registration Number': e.test['Registration Number'], 'Annual Turnover': p}).to_csv(path, index=False)
        validate_csv(pd.read_csv(path), e.test)
        assert np.max(np.abs(pd.read_csv(path)['Annual Turnover'] - p)) < .01
        np.savez_compressed(e.out / f'full_fit_trial_{i}.npz', prediction=p, baseline=b)
        change = float(np.sqrt(np.mean((p - b) ** 2)))
        check.update(trial=i, csv=str(path), csv_sha256=sha(path), choice=choice,
            prediction_rms_difference_from_v7=change, conditional_portal_rmse_lower_bound=max(0, baseline.PORTAL_RMSE - change))
        checks.append(check)
        log({'exported': str(path), 'conditional_portal_rmse_lower_bound': check['conditional_portal_rmse_lower_bound']})
    save(e.out / 'submission_checks.json', checks)
    if not decision['trials']:
        recommended = e.out.parent / 'recommended_submission'
        recommended.mkdir(exist_ok=True)
        for ext in ['csv', 'ipynb']:
            shutil.copyfile(ROOT / 'scored_v7_backup' / f'{STEM}.{ext}', recommended / f'{STEM}.{ext}')
    log({'export': 'complete', 'qualified_files': len(checks), 'v7_reproduction_max_difference': difference})


def export(e):
    import time
    state = e.cache.state()
    if state['seconds'] >= e.cache.hours * 3600:
        raise BudgetExceeded('No compute budget remains for fresh export.')
    started = time.monotonic()
    try:
        _export(e)
    finally:
        state = e.cache.state()
        elapsed = time.monotonic() - started
        state['seconds'] += elapsed
        state['by_stage']['export'] = state['by_stage'].get('export', 0) + elapsed
        state['maximum_hours'] = e.cache.hours
        state['measurement'] = 'Aggregate checkpoint-computation and fresh-export elapsed seconds; isolated notebook execution recorded separately.'
        save(e.cache.budget, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'pilot', 'screen', 'refine', 'confirm', 'export', 'all'])
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--budget-hours', type=float, default=18)
    args = parser.parse_args()
    assert 0 < args.budget_hours <= 18
    e = Engine(args.data_dir, args.out_dir, args.budget_hours)
    for stage in (['audit', 'pilot', 'screen', 'refine', 'confirm', 'export'] if args.stage == 'all' else [args.stage]):
        e.cache.stage = stage
        try:
            globals()[stage](e)
        except BudgetExceeded as error:
            save(e.out / 'budget_stop.json', {'stage': stage, 'reason': str(error), 'completed_candidates_only': True})
            if stage != 'export':
                save(e.out / 'decision.json', {'candidates': [], 'trials': [], 'decision': 'retain V7: research stopped at compute budget', 'portal_uploads_by_agent': 0})
                e.cache.stage = 'export'
                export(e)
                break
            raise
