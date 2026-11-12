import sys,json,logging
from pathlib import Path
logging.disable(logging.CRITICAL)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from colab_control import state,requests
s=state.get_session('v15-retry1-free-t4')
try:
    response=requests.get(s.url.rstrip('/')+'/api/kernels',params={'authuser':'0','colab-runtime-proxy-token':s.token},timeout=(10,30))
    response.raise_for_status()
    print(json.dumps({'kernels':[{'execution_state':k.get('execution_state'),'last_activity':k.get('last_activity'),'connections':k.get('connections'),'selected':k.get('id')==s.kernel_id} for k in response.json()]}),flush=True)
except Exception as error:print(json.dumps({'status':'unavailable','error_type':type(error).__name__}),flush=True)
