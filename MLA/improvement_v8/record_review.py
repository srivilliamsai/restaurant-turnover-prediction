"""Record static repository evidence and the user-reported portal observation.

Never import or execute the downloaded repository notebook.
"""
from decimal import Decimal
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
V7 = BASE / 'improvement_v7'
PUBLIC = Path('/tmp/restaurant_github_review_20261026')
STEM = 'submission'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rows(path):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        values = list(reader)
        return reader.fieldnames, values


if __name__ == '__main__':
    observation_path = V7 / 'portal_result_2026-10-26.json'
    observation = json.loads(observation_path.read_text())
    screenshot = Path(observation['screenshot'])
    try:
        observation['screenshot_sha256'] = sha(screenshot)
    except OSError:
        observation['screenshot_sha256'] = None
        observation['screenshot_evidence'] = 'Read from the image attached in the conversation; the macOS-protected source file could not be hashed.'
    scored = V7 / observation['csv']
    assert sha(scored) == observation['csv_sha256']
    previous = Decimal('12046722.3691139')
    current = Decimal('12024316.5355221')
    leader = Decimal('4796740.5600053')
    assert previous - current == Decimal(str(observation['rmse_reduction']))
    save(observation_path, observation)

    report = {'reviewed_on': '2026-10-26',
              'repository': 'https://github.com/imanasjoshi/restaurant_turnover_prediction',
              'ref_downloaded': 'master',
              'review_mode': 'Static JSON/code inspection only; no external notebook execution',
              'external_code_used_in_candidate': False,
              'external_predictions_used_in_candidate': False,
              'input_checks': {}}
    for name in ['Train_dataset.csv', 'Test_dataset.csv']:
        public_hash, official_hash = sha(PUBLIC / name), sha(BASE / name)
        assert public_hash == official_hash
        header, data = rows(PUBLIC / name)
        report['input_checks'][name] = {
            'sha256': public_hash, 'identical_to_official': True,
            'rows': len(data), 'columns': len(header)}
    book = json.loads((PUBLIC / 'Restaurant_Turnover_Improved_Submission.ipynb').read_text())
    code_cells = [c for c in book['cells'] if c['cell_type'] == 'code']
    report['notebook'] = {
        'sha256': sha(PUBLIC / 'Restaurant_Turnover_Improved_Submission.ipynb'),
        'code_cells': len(code_cells),
        'execution_counts': [c.get('execution_count') for c in code_cells],
        'saved_output_count': sum(len(c.get('outputs', [])) for c in code_cells),
        'runtime_reproduction_verified': False}
    scored_header, scored_rows = rows(scored)
    public_header, public_rows = rows(PUBLIC / 'submissions_improved.csv')
    assert public_header == scored_header == ['Registration Number', 'Annual Turnover']
    assert len(public_rows) == len(scored_rows) == 500
    assert [r['Registration Number'] for r in public_rows] == [r['Registration Number'] for r in scored_rows]
    assert len({r['Registration Number'] for r in public_rows}) == 500
    p = np.array([float(r['Annual Turnover']) for r in public_rows])
    baseline = np.array([float(r['Annual Turnover']) for r in scored_rows])
    assert np.isfinite(p).all() and (p >= 0).all()
    difference = float(np.sqrt(np.mean((p - baseline) ** 2)))
    report['public_csv_diagnostic'] = {
        'sha256': sha(PUBLIC / 'submissions_improved.csv'),
        'ids_match_scored_v7_and_order': True, 'rows': 500,
        'rms_difference_from_v7': difference,
        'conditional_portal_rmse_lower_bound': max(0., float(current) - difference),
        'actual_portal_rmse': None,
        'assumptions': 'Correct v7 Trial 1 score attribution; ordinary unweighted RMSE on all the same 500 restaurants',
        'hidden_targets_used': False,
        'can_reach_previously_shown_leader_under_assumptions': float(current) - difference <= float(leader)}
    report['static_findings'] = [
        'README claims about 19.83M local RMSE; no verified portal score supplied.',
        'Final stacker and affine calibration are scored on their fitting rows.',
        'Meta-level KFold reuses a global base-model OOF matrix rather than nesting the entire pipeline.',
        'Optional library fallback can change active models across environments.',
        'Native HGB and fold-model averaging are hypotheses for original bounded experiments.']
    report['current_best'] = observation
    report['gap_to_previously_shown_leader'] = {
        'leader_rmse': str(leader), 'current_gap': str(current - leader),
        'required_reduction_percent': str((current - leader) * 100 / current),
        'live_leaderboard_checked': False}
    report['v8_training_performed'] = False
    save(ROOT / 'github_review.json', report)

    manifest_path = V7 / 'delivery_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault('pre_portal_report_sha256', manifest['outputs']['REPORT.md'])
    manifest['outputs']['REPORT.md'] = sha(V7 / 'REPORT.md')
    manifest['outputs']['portal_result_2026-10-26.json'] = sha(observation_path)
    manifest['outputs']['portal_trials.csv'] = sha(V7 / 'portal_trials.csv')
    manifest['decision_context'] = 'Frozen pre-upload local selection; see portal_observations for subsequent user-reported results.'
    manifest['portal_observations'] = [observation]
    manifest['current_best_observed'] = {
        'version': 'v7_trial_1', 'portal_rmse': float(current),
        'csv_sha256': observation['csv_sha256'],
        'attribution': observation['attribution']}
    manifest['zip_context'] = 'V7_Delivery.zip is the original pre-upload snapshot, preserved unchanged.'
    save(manifest_path, manifest)
    print(json.dumps({
        'static_review_saved': str(ROOT / 'github_review.json'),
        'current_best_portal_rmse': str(current),
        'gain': str(previous - current),
        'github_public_csv_conditional_bound': float(current) - difference,
        'candidate_files_unchanged': True,
        'v8_models_fitted': 0}, indent=2))
