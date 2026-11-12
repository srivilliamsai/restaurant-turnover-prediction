"""Capture only public source text and release metadata; no model weights or training."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_sources'
def get(url):
 request=urllib.request.Request(url,headers={'User-Agent':'V18-research','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(request,timeout=40) as response:
  value=response.read(8_000_001)
 assert len(value)<=8_000_000,'Source exceeds planning download limit'
 return value
def js(url):return json.loads(get(url))
def capture(label,repo,files):
 info=js('https://api.github.com/repos/'+repo+'/commits/main')
 revision=info['sha']
 tree=js('https://api.github.com/repos/'+repo+'/git/trees/'+revision+'?recursive=1')
 names=[v['path'] for v in tree['tree'] if v['type']=='blob']
 chosen=[p for p in files if p in names]
 if label=='exaone':
  chosen += [p for p in names if p.startswith('src/') and p.endswith('.py') and any(v in p for v in ['regressor','regression','preprocessing','checkpoint','distribution'])]
 record={'repository':repo,'revision':revision,'commit_date':info['commit']['committer']['date'],'files':{}}
 for path in dict.fromkeys(chosen):
  url='https://raw.githubusercontent.com/'+repo+'/'+revision+'/'+path
  raw=get(url);target=OUT/label/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  record['files'][path]={'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 return record
records={}
for label,repo,files in [
 ('exaone','LGAI-Research/EXAONE-Tabular',['README.md','pyproject.toml','src/exaonetabular/config.py','src/exaonetabular/presets.py','src/exaonetabular/estimators.py']),
 ('tabpfn','PriorLabs/TabPFN',['examples/finetune_regressor.py','src/tabpfn/finetuning/finetuned_regressor.py','src/tabpfn/finetuning/finetuned_base.py','src/tabpfn/finetuning/train_util.py']),
 ('tabfm','google-research/tabfm',['README.md','requirements.txt'])]:
 records[label]=capture(label,repo,files)
print(json.dumps({k:{'revision':v['revision'],'source_files':len(v['files'])} for k,v in records.items()},indent=2))
(ROOT/'source_review_registry.json').write_text(json.dumps(records,indent=2)+'\n')
