import os,json,logging,uuid,time
from pathlib import Path
logging.disable(logging.CRITICAL)
os.environ.update(COLAB_CLI_ASSIGN_RETRIES='0',COLAB_CLI_REQUEST_TIMEOUT='60')
from colab_cli.common import state
from colab_cli.client import Variant,Accelerator,ColabRequestError,TooManyAssignmentsError
from colab_cli.state import SessionState
root=Path(__file__).resolve().parents[1]
try:
    res=state.client.assign(uuid.uuid4(),variant=Variant.GPU,accelerator=Accelerator.T4)
    info=res.runtime_proxy_info
    state.store.add(SessionState(name='v15-free-t4',token=info.token,url=info.url,endpoint=res.endpoint,token_expires_at=info.expires_at(),variant='GPU',accelerator='T4',machine_shape='STANDARD'))
    result={'allocated':True,'owned_session':'v15-free-t4','accelerator':'T4'}
except ColabRequestError as error:
    result={'allocated':False,'error_type':'ColabRequestError','http_status':error.response.status_code,'http_reason':error.response.reason,'request_method':error.request.method}
    body=(error.response_body or '').lower()
    result['recognized_signals']=[label for label in ['quota','capacity','usage limit','resource exhausted','permission denied','forbidden','not found','invalid','unavailable','too many','accelerator','gpu','authentication','internal error'] if label in body]
except TooManyAssignmentsError:result={'allocated':False,'http_status':412,'reason':'Allocation precondition refused'}
except Exception as error:result={'allocated':False,'error_type':type(error).__name__}
result['epoch']=time.time();(root/'transport_run/allocation_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
