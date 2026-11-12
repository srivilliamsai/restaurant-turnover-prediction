"""Read public checkpoint metadata and published source only; never download weights."""
import hashlib,io,json,tarfile,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def get(url,limit=12_000_000):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'V18-research'}),timeout=40) as r:value=r.read(limit+1)
 assert len(value)<=limit
 return value
record={}
url='https://huggingface.co/api/models/LG-AI-Research/EXAONE-Tabular?blobs=true'
m=json.loads(get(url))
n='exaone-tabular-regressor-v1_default.safetensors'
files=[v for v in m.get('siblings',[]) if v['rfilename']==n]
record['exaone']={'metadata_url':url,'repo_id':m['id'],'revision':m['sha'],'regression_files':files,'weights_downloaded':False}
p=json.loads(get('https://pypi.org/pypi/tabpfn/9.2.0/json'))
f=next((v for v in p['urls'] if v['packagetype']=='sdist'),p['urls'][0])
raw=get(f['url']);assert hashlib.sha256(raw).hexdigest()==f['digests']['sha256']
keep=['src/tabpfn/finetuning/finetuned_base.py','src/tabpfn/finetuning/finetuned_regressor.py','src/tabpfn/finetuning/train_util.py']
files={}
if f['filename'].endswith('.whl'):
 z=zipfile.ZipFile(io.BytesIO(raw));paths=z.namelist();reader=z.read
else:
 z=tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz');paths=z.getnames();reader=lambda name:z.extractfile(name).read()
for wanted in keep:
 suffix=wanted.replace('src/','')
 name=next(name for name in paths if name.endswith(suffix))
 b=reader(name);target=ROOT/'research_sources/tabpfn_9_2_0'/wanted
 target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 files[wanted]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
record['tabpfn_9_2_0']={'version':p['info']['version'],'archive_url':f['url'],'archive_sha256':hashlib.sha256(raw).hexdigest(),'files':files,'installed':False}
(ROOT/'release_review_registry.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
