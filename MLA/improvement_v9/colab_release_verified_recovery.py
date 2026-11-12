"""Save safe environment metadata and release only this run's idle owned T4."""
import hashlib
import json
import subprocess
from pathlib import Path
import requests
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime
root=Path(__file__).resolve().parent
run=root/'colab_cli_runs/20261026T160148Z'
checks=json.loads((run/'recovery_draft/fresh_execution_checks.json').read_text())
assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
assert hashlib.sha256((run/'recovery_draft/fresh_execution.csv').read_bytes()).hexdigest()==checks['csv_sha256']
book=json.loads((run/'recovery_draft/V9_Portable_Colab_executed.ipynb').read_text())
assert all(c['execution_count'] is not None for c in book['cells'] if c['cell_type']=='code')
session=state.get_session('v9-cli-t4');owned_endpoint=session.endpoint
runtime=ColabRuntime(session.url,session.token,kernel_id=session.kernel_id,session_id=session.session_id)
try:
    outputs=runtime.execute_code((root/'colab_freeze_environment.py').read_text(),timeout=180)
    for x in outputs:
        if x.get('output_type')=='stream':print(x.get('text',''),end='')
        if x.get('output_type')=='error':raise RuntimeError(x.get('ename','Error')+': '+x.get('evalue',''))
finally:runtime.stop()
session=state.get_session('v9-cli-t4');assert session.endpoint==owned_endpoint
ContentsClient(session).download('/content/v9_cli_20261026T160148Z/colab_environment_manifest.json',str(run/'colab_environment_manifest.json'))
response=requests.get(session.url.rstrip('/')+'/api/kernels',params={'authuser':'0','colab-runtime-proxy-token':session.token},timeout=30)
response.raise_for_status();kernels=response.json()
assert len(kernels)<=1 and all(k['execution_state']=='idle' for k in kernels),'Do not stop a busy runtime.'
assert state.get_session('v9-cli-t4').endpoint==owned_endpoint
subprocess.run([str(root/'colab_cli_env/bin/colab'),'--auth','oauth2','stop','-s','v9-cli-t4'],check=True,timeout=180)
(run/'runtime_cleanup.json').write_text(json.dumps({'owned_runtime_released':True,'session':'v9-cli-t4','gpu_outputs_backed_up':True,'paid_resources_requested':False,'portal_uploads':0},indent=2)+'\n')
print('Released the idle owned free T4; all required GPU outputs are saved locally.')
