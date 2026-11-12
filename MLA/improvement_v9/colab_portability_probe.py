"""Decompose the saved V7 export fit; no model is trained by this probe."""
import json
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime

root = Path(__file__).resolve().parent
run = root / 'colab_cli_runs' / '20261026T160148Z'
session = state.get_session('v9-cli-t4')
runtime = ColabRuntime(session.url, session.token, kernel_id=session.kernel_id,
                       session_id=session.session_id)
source = '''
import json, pickle, hashlib, platform
import legacy_reference as r, legacy_models as lm
from pathlib import Path
diagnostic_dir = WORK_ROOT / 'export_diagnostics'
fit_path = diagnostic_dir / 'fresh_v7_fit.pkl'
assert hashlib.sha256(fit_path.read_bytes()).hexdigest() == '59411e198c10e429ecbff616a0e739be781824967de90fd2af24fe544ec15d53'
with fit_path.open('rb') as f: saved_fit = pickle.load(f)
fresh_train, fresh_test, fresh_y = lm.load_data(DATA_DIR)
pd.testing.assert_frame_equal(e.train, fresh_train)
pd.testing.assert_frame_equal(e.test, fresh_test)
assert np.array_equal(e.y, fresh_y)
predictions = {}
for name, fitted in saved_fit['legacy']['v2'].items():
    predictions['v2__'+name] = r.predict_fitted(fitted, r.features(e.test, e.contract['v2_configs'][name]['variant']))
for index, fitted in enumerate(saved_fit['legacy']['v3']):
    predictions['v3__'+str(index)] = np.maximum(fitted.predict(r.features(e.test, 'base')), 0)
for name, fitted_list in saved_fit['legacy']['new'].items():
    for index, fitted in enumerate(fitted_list):
        predictions['new__'+name+'__'+str(index)] = r.predict_v5(fitted, e.test)
base_prediction = r.predict_final(e.test, saved_fit['legacy'], e.contract)
predictions['v5'] = base_prediction
for index, fitted in enumerate(saved_fit['models']):
    predictions['residual__'+str(index)] = lm.predict_component(fitted, e.test, base_prediction)
predictions['v7'] = lm.predict_submission(saved_fit, e.test)
np.savez_compressed(diagnostic_dir / 'cloud_components.npz', **predictions)
report = {'python': platform.python_version(), 'machine': platform.machine(),
          'system': platform.system(), 'train_and_test_unmodified': True,
          'feature_columns': r.features(e.train).columns.tolist(),
          'feature_frame_sha256': hashlib.sha256(pd.util.hash_pandas_object(r.features(e.train),index=True).values.tobytes()).hexdigest(),
          'target_sha256': hashlib.sha256(e.y.tobytes()).hexdigest(),
          'v3_params': saved_fit['legacy']['v3'][0].get_all_params(),
          'residual_params': saved_fit['models'][0]['model'].get_all_params()}
(diagnostic_dir / 'cloud_components.json').write_text(json.dumps(report,indent=2)+'\\n')
print('CLOUD_INPUTS_UNMODIFIED', True)
print('CLOUD_COMPONENTS_SAVED', len(predictions))
'''
try:
    outputs = runtime.execute_code(source, timeout=180)
    for item in outputs:
        if item.get('output_type') == 'stream': print(item.get('text',''), end='')
        if item.get('output_type') == 'error': raise RuntimeError(item.get('ename','Error')+': '+item.get('evalue',''))
    contents = ContentsClient(state.get_session('v9-cli-t4'))
    remote = '/content/v9_cli_20261026T160148Z/v9_fresh__h2_j7gd/export_diagnostics/'
    target = run / 'export_diagnostics'
    target.mkdir(exist_ok=True)
    for name in ['cloud_components.npz', 'cloud_components.json']:
        contents.download(remote+name,str(target/name))
    print('Downloaded two explicit component diagnostic files.')
finally:
    runtime.stop()
