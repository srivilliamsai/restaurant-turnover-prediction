import json
import subprocess
from pathlib import Path
root=Path('/content/mla_v10')
record=json.loads((root/'stage_pid.json').read_text())
for command in [
    ['ps','-p',str(record['pid']),'-o','pid,etime,pcpu,rss,stat'],
    ['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader']]:
    result=subprocess.run(command,capture_output=True,text=True,check=False)
    print(result.stdout or result.stderr)
path=root/'results/stage_status.json'
if path.exists():print(path.read_text())
