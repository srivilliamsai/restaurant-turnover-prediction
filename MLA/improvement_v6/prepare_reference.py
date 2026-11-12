"""Copy frozen function definitions, without importing side-effectful old scripts."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent


def extract(relative, functions=(), constants=()):
    path = BASE / relative
    source = path.read_text()
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            found[node.name] = ast.get_source_segment(source, node)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    found[target.id] = ast.get_source_segment(source, node)
    return [found[name] for name in [*constants, *functions]]


if __name__ == '__main__':
    header = '''"""Frozen v5 predictor, copied verbatim from the preserved implementations."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
'''
    pieces = [header]
    pieces += extract('improvement_v2/experiment.py',
                      ['features', 'rmse', 'sklearn_model', 'fit_predict', 'predict_fitted'],
                      ['TARGET', 'ID', 'EVENTS', 'SOCIAL', 'CORE'])
    pieces += extract('improvement_v3/experiments.py', ['cat_fit'])
    pieces += extract('improvement_v4/pipeline.py', ['get_splits', 'compute_v3', 'check_submission'])
    pieces += extract('improvement_v5/new_models.py', ['v5_features', 'inverse_target', 'fit_v5', 'predict_v5'])
    pieces += extract('improvement_v5/pipeline.py', ['evaluate', 'full_fit', 'predict_final'])
    (ROOT / 'reference.py').write_text('\n\n'.join(pieces) + '\n')
    source = BASE / 'improvement_v5/results/candidate_lock.json'
    (ROOT / 'v5_contract.json').write_bytes(source.read_bytes())
    sources = ['improvement_v2/experiment.py', 'improvement_v3/experiments.py',
               'improvement_v4/pipeline.py', 'improvement_v5/new_models.py', 'improvement_v5/pipeline.py']
    (ROOT / 'reference_provenance.json').write_text(json.dumps({
        'method': 'AST extraction of exact function source; no old module imports',
        'sources': {p: hashlib.sha256((BASE / p).read_bytes()).hexdigest() for p in sources},
        'reference_sha256': hashlib.sha256((ROOT / 'reference.py').read_bytes()).hexdigest()
    }, indent=2) + '\n')
