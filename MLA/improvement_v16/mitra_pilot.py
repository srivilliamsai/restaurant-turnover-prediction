"""Inspect the official Mitra adapter's fitted-model lifecycle before any accuracy test."""
import ast
import json
from core import ROOT, sha, save


def inspect_source(engine):
    root = ROOT / 'engineering_sources/mitra'
    identity = json.loads((root / 'source_identity.json').read_text())
    for name, record in identity['files'].items():
        assert sha(root / name.split('/')[-1]) == record['sha256']
    api = (root / 'api.py').read_text()
    runner = (root / 'runner.py').read_text()
    tree = ast.parse(api)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MitraFinetune')
    methods = sorted(node.name for node in model.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'))
    runner_tree = ast.parse(runner)
    run_view = next(node for node in runner_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_view')
    lifecycle = ast.get_source_segment(runner, run_view)
    prediction = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'predict')
    predict_source = ast.get_source_segment(api, prediction)
    assert 'self._execute_view(X_test)' in predict_source
    assert 'tempfile.TemporaryDirectory' in lifecycle
    assert 'child_main(str(work)' in lifecycle and 'return val_proba, test_proba' in lifecycle
    assert methods == ['fit', 'predict', 'predict_distribution', 'predict_proba'], 'Official API changed; a new lifecycle review is required'
    result = {'status': 'engineering_excluded', 'stage': 'supported-fitted-model-reuse',
              'reason': 'Official MitraFinetune.predict launches run_view for each query. run_view trains inside a TemporaryDirectory and returns arrays/distributions, then removes the fitted directory. The public API has no fitted-child reuse/load/predict interface. The predeclared persistent-inference gate cannot pass without changing the official adapter.',
              'repository': identity['repository'], 'revision': identity['revision'],
              'public_methods': methods, 'source_identity': identity,
              'api_predict_line': prediction.lineno, 'run_view_line': run_view.lineno,
              'full_recipe_executed': False, 'checkpoint_downloaded': False,
              'accuracy_tested': False, 'selection_uses_validation_rmse': False,
              'budget_changed': False, 'substitute_model_added': False}
    save(engine.out / 'mitra_engineering_review.json', result)
    return result
