"""Full outer-fold size feasibility and behavior checks before V9 screening."""
import json
import time
import importlib.util
from pathlib import Path
import numpy as np
import modeling as models


def run(e):
    from engine import save, log, sha, check_behavior
    a, b = e.splits(42)[0]
    families = {}
    for family, name in [('ebm', 'ebm_raw_i10'), ('tabpfn', 'tabpfn_raw_compact')]:
        if family == 'tabpfn':
            access = Path(__file__).resolve().parent / 'colab_access.json'
            info = json.loads(access.read_text()) if access.exists() else {}
            if not importlib.util.find_spec('torch') or not importlib.util.find_spec('tabpfn'):
                families[family] = {'eligible': False, 'status': 'not_run',
                    'reason': 'No local CUDA/TabPFN environment; remote Colab access outcome is recorded separately.',
                    'colab_access': info, 'training_rows_subsampled': False}
                continue
            import torch
            if not torch.cuda.is_available():
                families[family] = {'eligible': False, 'status': 'not_run', 'reason': 'No available CUDA GPU.'}
                continue
        start = time.monotonic()
        try:
            fitted = models.fit_model(e.train.iloc[a], e.y[a], models.SPECS[name])
            p = models.predict_model(fitted, e.train.iloc[b])
            test_p = models.predict_model(fitted, e.test)
            checks = check_behavior(lambda raw: models.predict_model(fitted, raw), e.test, test_p)
            repeated = models.fit_predict(e.train.iloc[a], e.y[a], e.train.iloc[b], models.SPECS[name])
            difference = float(np.max(np.abs(p - repeated)))
            assert difference < .01
            families[family] = {'eligible': True, 'status': 'passed', 'fit_rows': len(a), 'validation_rows': len(b),
                'settings': models.SPECS[name], 'repeat_difference_INR': difference, 'checks': checks,
                'no_outer_validation_labels_passed_to_fit': True}
            if family == 'tabpfn':
                families[family]['execution_contract'] = models.execution_contract()
        except Exception as error:
            families[family] = {'eligible': False, 'status': 'failed', 'error': repr(error)}
        finally:
            elapsed = time.monotonic() - start
            families[family]['seconds'] = elapsed
            state = e.cache.state(); state['seconds'] += elapsed
            state['by_stage']['pilot'] = state['by_stage'].get('pilot', 0) + elapsed
            state['maximum_hours'] = e.cache.hours
            save(e.cache.budget, state)
        log({'pilot': family, **families[family]})
    record = {'families': families, 'sources': e.sources, 'inputs': e.inputs, 'runtime': e.runtime,
              'planned_settings': len(models.SPECS), 'fresh_full_fold_fit': True}
    save(e.out / 'pilot.json', record)
    assert any(f['eligible'] for f in families.values()), 'No V9 family is feasible.'
