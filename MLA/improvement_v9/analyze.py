"""Original-target segment and fold diagnostics from every completed comparison."""
import json
import numpy as np
import pandas as pd
from engine import Engine, ROOT, sha, rmse, save


def run():
    e = Engine()
    table = pd.read_csv(e.out / 'screening.csv')
    segments, folds = [], []
    for scheme in [42, 71]:
        base = e.baseline(scheme)
        worst = np.argsort((e.y - base) ** 2)[-10:]
        components = {}
        for name in table.name.unique():
            p = np.zeros(len(e.y))
            for f, (_, indices) in enumerate(e.splits(scheme)):
                path = e.out / 'cache' / f'{name}_plain_{scheme}_fold{f}.npz'
                assert sha(path) == json.loads(path.with_suffix('.json').read_text())['sha256']
                p[indices] = np.load(path, allow_pickle=False)['prediction']
            components[name] = p
        for r in table.to_dict(orient='records'):
            p = (1 - r['weight']) * base + r['weight'] * components[r['name']]
            for label, mask in [('<=40M', e.y <= 40e6), ('40-80M', (e.y > 40e6) & (e.y <= 80e6)), ('>80M', e.y > 80e6)]:
                segments.append({'name': r['name'], 'weight': r['weight'], 'scheme': scheme,
                    'segment': label, 'rows': int(mask.sum()), 'v7_rmse': rmse(e.y[mask], base[mask]),
                    'candidate_rmse': rmse(e.y[mask], p[mask]), 'gain': rmse(e.y[mask], base[mask]) - rmse(e.y[mask], p[mask]),
                    'v7_sse': float(((e.y[mask] - base[mask]) ** 2).sum()),
                    'candidate_sse': float(((e.y[mask] - p[mask]) ** 2).sum())})
            for f, (_, indices) in enumerate(e.splits(scheme)):
                folds.append({'name': r['name'], 'weight': r['weight'], 'scheme': scheme, 'fold': f,
                    'rows': len(indices), 'tail_above80M': int((e.y[indices] > 80e6).sum()),
                    'v7_rmse': rmse(e.y[indices], base[indices]), 'candidate_rmse': rmse(e.y[indices], p[indices])})
    pd.DataFrame(segments).to_csv(e.out / 'development_segments.csv', index=False)
    pd.DataFrame(folds).to_csv(e.out / 'development_folds.csv', index=False)
    print(f'Saved all {len(segments)} segment and {len(folds)} fold diagnostics.', flush=True)


if __name__ == '__main__': run()
