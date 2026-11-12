"""Read-only owned-runtime workload probe; prints no authenticated endpoints."""
import base64,json,logging,time
from urllib.parse import urlparse
from colab_cli.common import state
from colab_cli.contents import ContentsClient
import websocket
import colab_control as c
logging.disable(logging.CRITICAL)
code="""import json,psutil,subprocess,time
from pathlib import Path
processes=[]
for p in psutil.process_iter(['pid','name','cmdline','memory_info','cpu_times','create_time']):
 try:
  v=p.info
  if any('pilot_worker.py' in z for z in (v['cmdline'] or [])):
   processes.append({'pid':v['pid'],'name':v['name'],'resident_bytes':v['memory_info'].rss,'cpu_seconds':v['cpu_times'].user+v['cpu_times'].system,'elapsed_seconds':time.time()-v['create_time']})
 except psutil.Error:pass
q=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
Path('/content/mla_v17/pilot_runtime_probe.json').write_text(json.dumps({'epoch':time.time(),'pilot_processes':processes,'gpu':q.stdout.strip()}))
"""
s=state.get_session(c.SESSION)
u=urlparse(s.url)
ws=None
try:
 ws=websocket.create_connection(('wss' if u.scheme=='https' else 'ws')+'://'+u.netloc+'/colab/tty?colab-runtime-proxy-token='+s.token,timeout=15)
 command='python -c '+__import__('shlex').quote('exec('+repr(code)+')')+'\n'
 ws.send(json.dumps({'data':command}))
 deadline=time.monotonic()+20
 content=ContentsClient(s)
 while time.monotonic()<deadline:
  try:
   r=json.loads(c.read(content,c.REMOTE+'/pilot_runtime_probe.json'))
   if time.time()-r['epoch']<20:
    c.put(c.ROOT/'transport_run/pilot_runtime_probe.json',json.dumps(r,indent=2).encode())
    print(json.dumps(r,indent=2));break
  except FileNotFoundError:pass
  time.sleep(1)
 else:print(json.dumps({'probe':'no fresh result'}))
except Exception as error:print(json.dumps({'probe_error_type':type(error).__name__}))
finally:
 if ws:ws.close()
