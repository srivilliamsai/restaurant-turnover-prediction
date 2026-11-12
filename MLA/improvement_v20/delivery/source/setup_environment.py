"""Reproducible frozen controls and two minimal candidate virtual environments."""
import hashlib, importlib.metadata, json, os, subprocess, sys, time, tarfile, urllib.request
from pathlib import Path
ROOT=Path(os.environ.get('MLA_V20_SETUP_ROOT','/content/mla_v20'))
ROOT.mkdir(exist_ok=True)
started=time.time();records={}

def run(name,command,timeout=1200):
    begin=time.time()
    with (ROOT/'setup.log').open('a') as log:
        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    records[name]={'returncode':p.returncode,'seconds':time.time()-begin}
    (ROOT/'setup_status.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'setup_step':name,**records[name]}),flush=True)
    assert p.returncode==0,'Environment step failed: '+name

control=json.loads((ROOT/'runtime_lock.json').read_text())['baseline']
run('frozen_control_dependencies',[sys.executable,'-m','pip','install','-q',*[n+'=='+v.split('+')[0] for n,v in control.items()]],1500)
minimal={n:v for n,v in control.items() if n in ['torch','numpy','pandas','scikit-learn','scipy','catboost','pyarrow','psutil','safetensors','huggingface-hub','joblib','packaging','tqdm','typing-extensions','triton']}
minimal['tabicl']='2.2.0'
commit='2b1696f4fdc22004cc757c44703a69cf326c0a11'
source=ROOT/'engineering_sources'/('structured-data-models-'+commit)
if not source.is_dir():
    source.parent.mkdir(exist_ok=True)
    archive=ROOT/'kumo_source.tar.gz'
    with urllib.request.urlopen('https://codeload.github.com/NVIDIA/structured-data-models/tar.gz/'+commit,timeout=90) as response:
        archive.write_bytes(response.read())
    with tarfile.open(archive) as bundle:bundle.extractall(source.parent,filter='data')
assert source.is_dir()
locked=json.loads((ROOT/'candidate_dependency_locks.json').read_text()) if (ROOT/'candidate_dependency_locks.json').exists() else {}
locks={}
for family in ['tabicl','kumo']:
    folder=ROOT/(family+'_env')
    run(family+'_venv',[sys.executable,'-m','venv','--without-pip',str(folder)],120)
    python=str(folder/'bin/python')
    req=dict(minimal)
    if family=='kumo':req.pop('tabicl')
    if family in locked:
        req={n:v for n,v in locked[family].items() if n.lower() not in ['pip','setuptools','structured-data-models']}
    pip=[sys.executable,'-m','pip','--python',python]
    run(family+'_dependencies',[*pip,'install','-q',*[n+'=='+v.split('+')[0] for n,v in req.items()]],1500)
    if family=='kumo':run('kumo_pinned_source',[*pip,'install','-q','--no-deps',str(source)],180)
    probe="import importlib.metadata,importlib.util,json,sys,torch;print(json.dumps({'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'cudf_discoverable':importlib.util.find_spec('cudf') is not None,'numba_discoverable':importlib.util.find_spec('numba') is not None,'system_site_packages':__import__('site').ENABLE_USER_SITE}))"
    records[family]=json.loads(subprocess.check_output([python,'-c',probe],text=True,timeout=90))
    assert not records[family]['cudf_discoverable'] and not records[family]['numba_discoverable']
    assert records[family]['gpu']=='Tesla T4' and records[family]['torch']=='2.11.0+cu130'
    locks[family]=records[family]['packages']
records['setup_complete']=True;records['seconds']=time.time()-started
(ROOT/'setup_status.json').write_text(json.dumps(records,indent=2)+'\n')
(ROOT/'candidate_dependency_locks.json').write_text(json.dumps(locks,indent=2)+'\n')
print('V20 clean candidate environments ready.',flush=True)
