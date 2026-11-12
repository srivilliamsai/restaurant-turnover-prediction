import sys,json,logging
from pathlib import Path
logging.disable(logging.CRITICAL)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from colab_control import state,ContentsClient,REMOTE
client=ContentsClient(state.get_session('v15-free-t4'))
for folder in [REMOTE+'/model_weights','/root/.cache/huggingface/hub/models--Prior-Labs--tabpfn_3_5/blobs','/root/.cache/huggingface/hub/models--Prior-Labs--TabPFN-v2-reg/blobs']:
    try:
        obj=client.list_dir(folder)
        print(json.dumps({'folder':folder,'files':[{'name':x.get('name'),'size':x.get('size'),'modified':x.get('last_modified')} for x in obj.get('content',[])]}),flush=True)
    except Exception as error:print(json.dumps({'folder':folder,'status':type(error).__name__}),flush=True)
