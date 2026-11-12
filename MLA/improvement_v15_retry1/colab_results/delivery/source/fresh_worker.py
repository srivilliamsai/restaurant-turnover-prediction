"""Fresh standalone inference; expected numerical outputs are read after prediction."""
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import baseline
import modeling
import verify
from core import ROOT, STEM, Engine, sha, save, validate_csv


def main():
    started = time.monotonic()
    manifest = json.loads((ROOT / 'standalone_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if name != 'reference/expected.csv':
            assert sha(ROOT / name) == expected, name
    out = Path(os.environ['MLA_V15_OUTPUT_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    e = Engine(os.environ['MLA_DATA_DIR'], out / 'fresh_context')
    assert e.inputs == manifest['inputs']
    with (ROOT / 'reference/verified_v7.pkl').open('rb') as handle:
        original = pickle.load(handle)
    v7 = baseline.predict(original, e.test)
    v7_difference = float(np.max(np.abs(v7 - pd.read_csv(ROOT / 'reference/scored_v7.csv')['Annual Turnover'].to_numpy(float))))
    assert v7_difference < .01
    base_model = modeling.fit(e.train, e.y, 'BASE')
    selected = old_p35 = None
    try:
        t9 = modeling.predict(base_model, e.test)
        v9 = .5 * v7 + .5 * t9
        v9_difference = float(np.max(np.abs(v9 - pd.read_csv(ROOT / 'reference/scored_v9.csv')['Annual Turnover'].to_numpy(float))))
        assert v9_difference < .01
        old_p35 = modeling.fit(e.train, e.y, 'P35_8')
        p35 = modeling.predict(old_p35, e.test)
        v14 = .25 * v9 + .75 * p35
        expected_v14 = pd.read_csv(ROOT / 'reference/scored_v14.csv')['Annual Turnover'].to_numpy(float)
        v14_difference = float(np.max(np.abs(v14 - expected_v14)))
        assert v14_difference < .01
        choice = manifest['choice']
        selected = modeling.fit(e.train, e.y, choice['name'])
        component = modeling.predict(selected, e.test)
        prediction = modeling.combine(v9, component, choice['weight'], p35)
        def predict_any(raw):
            old_v7 = baseline.predict(original, raw)
            old_t9 = modeling.predict(base_model, raw)
            frozen = .5 * old_v7 + .5 * old_t9
            return modeling.combine(frozen, modeling.predict(selected, raw), choice['weight'], modeling.predict(old_p35, raw))
        behavior = verify.inference_checks(predict_any, e.test)
        target_metadata = selected['target_metadata']
    finally:
        modeling.release(base_model)
        modeling.release(selected)
        modeling.release(old_p35)
    assert sha(ROOT / 'reference/expected.csv') == manifest['files']['reference/expected.csv']
    expected = pd.read_csv(ROOT / 'reference/expected.csv')['Annual Turnover'].to_numpy(float)
    difference = float(np.max(np.abs(prediction - expected)))
    assert difference < .01, f'Fresh reproduction differs INR {difference}'
    checks = verify.csv_checks(e, prediction)
    frame = pd.DataFrame({'Registration Number': e.test['Registration Number'], 'Annual Turnover': prediction})
    validate_csv(frame, e.test)
    path = out / (STEM + '.csv')
    frame.to_csv(path, index=False)
    validate_csv(pd.read_csv(path), e.test)
    assert len(path.read_text().splitlines()) == 501
    save(out / 'fresh_execution_checks.json', {
        'status': 'PASS', 'choice': choice, 'inputs': e.inputs,
        'v7_portable_difference_INR': v7_difference, 'fresh_v9_difference_INR': v9_difference,
        'fresh_v14_difference_INR': v14_difference, 'maximum_final_difference_INR': difference, 'full_blend_behavior': behavior,
        'target_metadata': target_metadata, 'csv_checks': checks, 'csv_sha256': sha(path),
        'seconds': time.monotonic() - started, 'new_tabpfn_components_freshly_fitted': True,
        'v7_freshly_retrained_in_colab': False, 'portal_uploads': 0})
    print('Fresh V15 execution PASS; maximum CSV difference INR', difference, flush=True)


if __name__ == '__main__':
    main()
