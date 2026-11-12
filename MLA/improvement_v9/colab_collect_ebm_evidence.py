"""Download only the 40 declared EBM prediction checkpoints and metadata."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient
root=Path(__file__).resolve().parent
target=root/'colab_cli_runs/20261026T160148Z/partial_backup/results/cache'
remote='/content/v9_cli_20261026T160148Z/v9_fresh__h2_j7gd/results/cache/'
session=state.get_session('v9-cli-t4')
stems=[f'ebm_{power}_i{interaction}_plain_{scheme}_fold{fold}'
       for power in ['raw','sqrt'] for interaction in [0,10]
       for scheme in [42,71] for fold in range(5)]
def download(stem):
    contents=ContentsClient(session)
    for extension in ['json','npz']:
        name=stem+'.'+extension
        contents.download(remote+name,str(target/name))
    path=target/(stem+'.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads(path.with_suffix('.json').read_text())['sha256']
    return stem
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    completed=list(executor.map(download,stems))
(target.parent/'ebm_backup.json').write_text(json.dumps({'completed':completed,'read_only_remote':True},indent=2)+'\n')
print('Backed up exactly',len(completed),'declared EBM checkpoint pairs.')
