"""Reproduce the scored restaurant turnover predictions on the original GPU setup."""

import argparse
import json
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

from check_submission import ID, MANIFEST, ROOT, TARGET, check_predictions, sha256, verify_hashes

TOLERANCE_INR = 0.01


def run(data, output):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
        os.environ[key] = '2'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

    import numpy as np
    import pandas as pd
    import torch

    import baseline
    import legacy_models
    import modeling
    from verify import inference_checks

    if not (torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Tesla T4'
            and torch.__version__ == '2.11.0+cu130'):
        raise RuntimeError('Reproduction requires a Tesla T4 with PyTorch 2.11.0+cu130.')

    manifest = json.loads(MANIFEST.read_text())
    verify_hashes(data, manifest['inputs'])
    model_path = ROOT / 'artifacts/verified_v7.pkl'
    verify_hashes(ROOT, {'artifacts/verified_v7.pkl': manifest['artifacts']['artifacts/verified_v7.pkl']})
    train, test, y = legacy_models.load_data(data)
    with model_path.open('rb') as handle:
        fitted_v7 = pickle.load(handle)

    def check_reference(prediction, name):
        verify_hashes(ROOT, {name: manifest['artifacts'][name]})
        expected = pd.read_csv(ROOT / name)[TARGET].to_numpy(float)
        difference = float(np.max(np.abs(prediction - expected)))
        if not difference < TOLERANCE_INR:
            raise ValueError(f'{name} differs by INR {difference}, expected less than INR {TOLERANCE_INR}.')
        return difference

    v7 = baseline.predict(fitted_v7, test)
    v7_error = check_reference(v7, 'artifacts/scored_v7.csv')
    fitted_v2 = None
    fitted_v35 = None
    try:
        fitted_v2 = modeling.fit(train, y, 'BASE', manifest['checkpoints']['BASE'])
        v9 = 0.5 * v7 + 0.5 * modeling.predict(fitted_v2, test)
        v9_error = check_reference(v9, 'artifacts/scored_v9.csv')
        fitted_v35 = modeling.fit(train, y, 'P35_8', manifest['checkpoints']['P35_8'])
        prediction = 0.25 * v9 + 0.75 * modeling.predict(fitted_v35, test)

        def predict(raw):
            v9 = 0.5 * baseline.predict(fitted_v7, raw) + 0.5 * modeling.predict(fitted_v2, raw)
            return 0.25 * v9 + 0.75 * modeling.predict(fitted_v35, raw)

        behavior = inference_checks(predict, test)
    finally:
        modeling.release(fitted_v2)
        modeling.release(fitted_v35)

    final_error = check_reference(prediction, 'submission.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        candidate = Path(temp) / 'submission.csv'
        pd.DataFrame({ID: test[ID], TARGET: prediction}).to_csv(candidate, index=False)
        rows = check_predictions(candidate, data)
        candidate.replace(output)

    print(json.dumps({
        'status': 'PASS', 'rows': rows, 'sha256': sha256(output),
        'v7_max_error_INR': v7_error, 'v9_max_error_INR': v9_error,
        'final_max_error_INR': final_error, 'inference_checks': behavior,
    }, indent=2))
    print(f'Wrote {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/submission.csv')
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads(MANIFEST.read_text())
    data = args.data_dir.resolve()
    protected = {ROOT / name for name in manifest['artifacts']}
    protected.update(data / name for name in manifest['inputs'])
    if output in protected or output == MANIFEST:
        parser.error('Choose an output path outside the source data and reference artifacts.')
    run(data, output)


if __name__ == '__main__':
    main()
