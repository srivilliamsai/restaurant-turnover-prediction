"""Download public weights only; no local dataset or artifact is transmitted."""
import hashlib,json,time
from pathlib import Path
import requests
root=Path(__file__).resolve().parent
record=json.loads((root/'research_sources/checkpoint_manifest.json').read_text())
folder=root/'weights';folder.mkdir(exist_ok=True)
for name in ['LICENSE','LimiX-2.ckpt']:
    want=next(x for x in record['files'] if x['rfilename']==name)
    path=folder/name
    if not path.exists():
        response=requests.get(f'https://huggingface.co/{record["repo_id"]}/resolve/{record["revision"]}/{name}',stream=True,timeout=(10,60));response.raise_for_status()
        temp=path.with_name(name+'.partial');count=0
        with temp.open('wb') as out:
            for block in response.iter_content(8*1024*1024):out.write(block);count+=len(block)
        assert count==want['size'];temp.replace(path)
    if 'lfs' in want:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==want['lfs']['sha256']
        print(json.dumps({'checkpoint':'verified','bytes':path.stat().st_size,'sha256':h.hexdigest()}),flush=True)
