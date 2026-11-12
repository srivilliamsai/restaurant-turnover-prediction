"""Prepare only the owned V20 runtime; never print authenticated connection data."""
import json,zipfile
from pathlib import Path
import colab_control as control
from colab_cli.common import state
from colab_cli.contents import ContentsClient
ROOT=Path(__file__).resolve().parent

def main():
    contents=ContentsClient(state.get_session(control.SESSION))
    archive=ROOT/'bootstrap_sources.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['setup_environment.py','runtime_lock.json','checkpoint_registry.json','source_registry.json','authorization.json','candidate_dependency_locks.json']:
            z.write(ROOT/name,name)
        for p in (ROOT/'engineering_sources').rglob('*'):
            if p.is_file() and p.suffix in ['.py','.json','.toml','.md']:
                z.write(p,str(p.relative_to(ROOT)))
    contents.upload(str(archive),control.REMOTE+'/bootstrap_sources.zip')
    code="""import os,sys,json,subprocess,time,zipfile
from pathlib import Path
root=Path('/content/mla_v20')
with zipfile.ZipFile(root/'bootstrap_sources.zip') as z:
    for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
    z.extractall(root)
auth=json.loads((root/'authorization.json').read_text())
env=os.environ.copy();env['MLA_V20_SETUP_ROOT']=str(root)
allowance=auth['engineering_stop_epoch']-time.time()-120
assert allowance>0
p=subprocess.run([sys.executable,'-u',str(root/'setup_environment.py')],env=env,timeout=allowance)
assert p.returncode==0,'Clean environment installation failed; inspect setup.log'
"""
    control.execute(code,3000)
    for name in ['setup_status.json','setup.log','candidate_dependency_locks.json']:
        control.put(ROOT/name,control.read(contents,control.REMOTE+'/'+name))
    control.execute("""import json,subprocess,sys,hashlib,zipfile
from pathlib import Path
from huggingface_hub import HfApi
root=Path('/content/mla_v20')
script='''import json,inspect,hashlib,importlib.metadata
from pathlib import Path
import tabicl
from tabicl import TabICLRegressor
package=Path(tabicl.__file__).parent
print(json.dumps({'module':TabICLRegressor.__module__,'class_file_sha256':hashlib.sha256(Path(inspect.getfile(TabICLRegressor)).read_bytes()).hexdigest(),'package_init_sha256':hashlib.sha256(Path(tabicl.__file__).read_bytes()).hexdigest(),'version':importlib.metadata.version('tabicl'),'package_files':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.py')}}))
'''
record=json.loads(subprocess.check_output([str(root/'tabicl_env/bin/python'),'-c',script],text=True,timeout=60))
assert record['version']=='2.2.0'
old=json.loads((root/'source_registry.json').read_text())
modules={'I8':{record['module']:record['class_file_sha256'],'tabicl':record['package_init_sha256']},'KS8-CLEAN':old['model_modules'].get('KS8-CLEAN',old['model_modules'].get('KS8'))}
registry={'model_modules':modules,'TabICL_package':record,'Kumo_source_commit':'2b1696f4fdc22004cc757c44703a69cf326c0a11','source_patches':False}
(root/'source_registry.json').write_text(json.dumps(registry,indent=2)+'\\n')
existing=json.loads((root/'checkpoint_registry.json').read_text())
revision=existing['models'].get('I8',{}).get('model_revision')
info=HfApi(token=False).model_info('jingang/TabICL',revision=revision,files_metadata=True)
file=next(x for x in info.siblings if x.rfilename=='tabicl-regressor-v2-20260212.ckpt')
assert file.lfs is not None
irec={'model_repository':info.id,'model_revision':info.sha,'filename':file.rfilename,'advertised_bytes':file.size,'advertised_sha256':file.lfs.sha256,'source_repository':'soda-inria/tabicl','package_version_or_source':'tabicl==2.2.0','weights_downloaded':False,'downloaded_bytes_verified':False}
oldcp=json.loads((root/'checkpoint_registry.json').read_text())
cp={'status':'immutable_metadata_locked_before_pilot','models':{'I8':irec,'KS8-CLEAN':oldcp['models'].get('KS8-CLEAN',oldcp['models'].get('KS8'))}}
(root/'checkpoint_registry.json').write_text(json.dumps(cp,indent=2)+'\\n')
with zipfile.ZipFile(root/'tabicl_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
    pkg=root/'tabicl_env/lib'
    for p in pkg.glob('python*/site-packages/tabicl/**/*.py'):z.write(p,'tabicl/'+str(p).split('/site-packages/tabicl/',1)[1])
print('Exact TabICL package source and public checkpoint metadata locked before any V20 accuracy tests.',flush=True)
""",180)
    for name in ['source_registry.json','checkpoint_registry.json','tabicl_source_snapshot.zip']:
        control.put(ROOT/name,control.read(contents,control.REMOTE+'/'+name))
    print('V20 candidate environments and immutable metadata prepared.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('V20 bootstrap failed:',type(error).__name__,'; inspect owned-runtime logs; authenticated details suppressed.',flush=True)
        raise SystemExit(1)
