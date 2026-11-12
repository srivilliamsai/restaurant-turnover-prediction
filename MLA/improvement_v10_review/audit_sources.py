"""Read-only source/data audit. Never imports or executes an attached script."""
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SUFFIXES = """1793041023703 1793041021982 1793041016975 1793041013037
1793041010817 1793040786727 1793040782819 1793040565406 1793040523135
1793040507412 1793040501263 1793040449983 1793040361521 1793040358474
1793040357375 1793039902514 1793039897406 1793039887567 1793039828263
1793039347510 1793039118315 1793039009620 1793038993172 1793038991194
1793038947785 1793038928480 1793038925753 1793038921109 1793038917092""".split()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources, groups = [], defaultdict(list)
    for suffix in SUFFIXES:
        path = Path('~/Downloads') / f'gemini-code-{suffix}.py'
        source = path.read_text()
        entry = dict(path=str(path), sha256=sha(path), lines=len(source.splitlines()))
        try:
            tree = ast.parse(source, filename=str(path))
            entry['syntax'] = 'valid Python syntax; not executed'
            entry['functions'] = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            entry['imports'] = sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                                      {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
        except SyntaxError as exc:
            entry['syntax'] = f'{exc.msg}, line {exc.lineno}'
        sources.append(entry)
        groups[entry['sha256']].append(path.name)
    train = pd.read_csv(ROOT / 'Train_dataset.csv')
    test = pd.read_csv(ROOT / 'Test_dataset.csv')
    v7_aliases = {'bengaluru': 'bangalore', 'banglore': 'bangalore', 'gurugram': 'gurgaon',
                  'gurgoan': 'gurgaon', 'bombay': 'mumbai', 'calcutta': 'kolkata', 'delhi': 'new delhi',
                  'bhubneshwar': 'bhubaneswar', 'hderabad': 'hyderabad', 'punr': 'pune'}
    # Extract only a literal dictionary for comparison, without executing its function.
    feature_path = Path('~/Downloads/gemini-code-1793040565406.py')
    tree = ast.parse(feature_path.read_text())
    city_func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'clean_city')
    assign = next(n for n in city_func.body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == 'mapping' for t in n.targets))
    attached_map = ast.literal_eval(assign.value)
    # Broad metropolitan pooling and ambiguous tokens are not spelling corrections.
    deferred = {'asifabadbanglore', 'ncr', 'navi mumbai', 'thane', 'secunderabad',
                'greater noida', 'sahibabad', 'shahibabad', ' trivandrum'}
    candidate_map = {**v7_aliases, **{k: v for k, v in attached_map.items() if k not in deferred}}

    def normalize(s):
        return s.astype('string').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True).fillna('Missing')

    city = {}
    train_base = normalize(train['City']).replace(v7_aliases)
    train_candidate = normalize(train['City']).replace(candidate_map)
    for name, frame in [('train', train), ('test', test)]:
        normalized = normalize(frame['City'])
        base = normalized.replace(v7_aliases)
        candidate = normalized.replace(candidate_map)
        changed = base != candidate
        city[name] = dict(raw_categories=int(frame['City'].nunique()), v7_categories=int(base.nunique()),
                         proposed_literal_map_categories=int(candidate.nunique()),
                         changed_rows=int(changed.sum()),
                         changed_tokens=normalized[changed].value_counts().to_dict(),
                         v7_unseen_rows=int((~base.isin(train_base.unique())).sum()),
                         proposed_unseen_rows=int((~candidate.isin(train_candidate.unique())).sum()))
    ratings = {}
    upper_bounds = {'Restaurant Zomato Rating': 5, 'Staff Responsivness': 8,
                    'Value for Money': 7, 'Hygiene Rating': 10, 'Food Rating': 10,
                    'Overall Restaurant Rating': 10}
    dates, turnover = {}, {}
    for name, frame in [('train', train), ('test', test)]:
        ratings[name] = {c: dict(min=float(frame[c].min()), max=float(frame[c].max()),
                                 zero_rows=int(frame[c].eq(0).sum()), missing_rows=int(frame[c].isna().sum()),
                                 outside_documented_range=int(((frame[c] < 1) | (frame[c] > maximum)).sum()))
                         for c, maximum in upper_bounds.items()}
        dt = pd.to_datetime(frame['Opening Day of Restaurant'], format='%d-%m-%Y', errors='coerce')
        dates[name] = dict(first=str(dt.min()), last=str(dt.max()), invalid_rows=int(dt.isna().sum()),
                           after_2015=int((dt > pd.Timestamp('2015-01-01')).sum()))
    target = train['Annual Turnover'].to_numpy(float)
    for step in [100000, 500000]:
        turnover[f'multiples_{step}'] = int(np.isclose(np.remainder(target, step), 0, atol=.001).sum())
    turnover.update(rows=len(target), above_80m=int((target > 80e6).sum()),
                    maximum=float(target.max()), unique=int(len(np.unique(target))))
    protected = ROOT / 'improvement_v9/scored_v9_backup/submission.csv'
    assert sha(protected) == '5a384dc3d8bf58c4e29f0d6cac646aa33afe22bdcdd186cccd9758342769bbd4'
    protected_notebook = protected.with_suffix('.ipynb')
    assert sha(protected_notebook) == '38a39003d0d7fb805b8725beac9ae80933929f6782c412166deef4d4df2f6973'
    result = dict(scope='Static review and read-only official-data diagnostics; no model training or portal upload.',
                  source_count=len(sources), unique_sources=len(groups),
                  duplicates=[v for v in groups.values() if len(v) > 1], sources=sources,
                  inputs={name: sha(ROOT / name) for name in ['Train_dataset.csv', 'Test_dataset.csv',
                                                            'Data_Dictionary.csv', 'Sample_Solution.csv']},
                  preserved_v9=dict(path=str(protected), sha256=sha(protected),
                                    notebook_path=str(protected_notebook), notebook_sha256=sha(protected_notebook),
                                    portal_rmse=11904515.9864996, screenshot_rank=9),
                  city=city, proposed_city_map=candidate_map, deferred_city_tokens=sorted(deferred),
                  ratings=ratings, dates=dates, turnover=turnover)
    (OUT / 'source_and_data_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['sources', 'proposed_city_map']}, indent=2))


if __name__ == '__main__':
    main()
