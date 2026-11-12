import json,subprocess,os,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cli=root.parent/'improvement_v9/colab_cli_env/bin/colab'
env=os.environ.copy();env.update(COLAB_CLI_ASSIGN_RETRIES='1',COLAB_CLI_RETRY_BACKOFF='5')
try:
    p=subprocess.run([str(cli),'--auth','oauth2','new','-s','v15-free-t4','--gpu','T4'],capture_output=True,text=True,timeout=180,env=env)
    text=p.stdout+p.stderr
    if p.returncode==0:result={'allocated':True,'owned_session':'v15-free-t4','accelerator':'T4'}
    elif 'Allocation refused (precondition failed)' in text:result={'allocated':False,'reason':'Colab refused allocation: temporary capacity/usage limit or assignment precondition (HTTP 412).'}
    elif 'Backend rejected accelerator' in text:result={'allocated':False,'reason':'Colab rejected T4 entitlement/quota (HTTP 400).'}
    else:
        reason='Colab CLI allocation failed; no authentication details emitted.'
        if 'ConnectTimeout' in text or 'ReadTimeout' in text:reason='Colab allocation request timed out.'
        if 'ConnectionError' in text:reason='Colab allocation encountered a connection error.'
        if 'ColabRequestError' in text:reason='Colab allocation returned a backend request error.'
        result={'allocated':False,'reason':reason,'returncode':p.returncode}
except subprocess.TimeoutExpired:result={'allocated':False,'reason':'Colab allocation did not finish within the bounded 180-second timeout.'}
result['epoch']=time.time();(root/'transport_run/allocation_retry.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
