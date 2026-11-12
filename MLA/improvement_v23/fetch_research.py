"""Fetch immutable public author sources; never send restaurant data."""
import concurrent.futures, hashlib, io, json, tarfile
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
REV='c76fa21bbb25fddca649e75a52dc9b1df66a659c'
def fetch(url):
    response=requests.get(url,timeout=(10,90)); response.raise_for_status(); return response.content
raw=fetch(f'https://api.github.com/repos/limix-ldm-ai/LimiX/tarball/{REV}')
folder=ROOT/'research_sources/LimiX'; folder.mkdir(parents=True,exist_ok=True)
files={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
    for member in archive:
        rel=Path(*Path(member.name).parts[1:])
        if not member.isfile() or rel.is_absolute() or '..' in rel.parts: continue
        if rel.suffix not in ['.py','.json','.md','.toml','.txt','.yml']: continue
        target=folder/rel; target.parent.mkdir(parents=True,exist_ok=True)
        data=archive.extractfile(member).read(); target.write_bytes(data)
        files[str(rel)]=hashlib.sha256(data).hexdigest()
(ROOT/'research_sources/source_manifest.json').write_text(json.dumps({'repository':'https://github.com/limix-ldm-ai/LimiX','revision':REV,'archive_sha256':hashlib.sha256(raw).hexdigest(),'files_sha256':files},indent=2)+'\n')
print(json.dumps({'revision':REV,'source_files':len(files),'archive_bytes':len(raw)}))
