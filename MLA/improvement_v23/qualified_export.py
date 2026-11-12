"""Only a qualified decision may produce a freshly executed V23 notebook."""
import base64,copy,json,os,shutil,sys,time,zlib
from pathlib import Path
import numpy as np
import pandas as pd
import experiment as e
ROOT=e.ROOT
STEM='submission'

def build_notebook(expected):
    import nbformat
    baseline=nbformat.read(ROOT/'reference/scored_v14.ipynb',as_version=4)
    bundle={}
    for name in ['adapter.py','protocol.json','research_sources/checkpoint_manifest.json','research_sources/source_manifest.json']:
        bundle[name]=base64.b64encode((ROOT/name).read_bytes()).decode()
    for p in (ROOT/'research_sources/LimiX').rglob('*'):
        if p.is_file() and p.suffix in ['.py','.json','.toml','.md','.txt','.yml']:
            bundle[str(p.relative_to(ROOT))]=base64.b64encode(p.read_bytes()).decode()
    packed=base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode()
    witness=base64.b64encode(np.asarray(expected,dtype='<f8').tobytes()).decode()
    runtime=e.read(ROOT/'resolved.json');audit=e.read(ROOT/'history_audit.json')
    book=nbformat.v4.new_notebook()
    book.cells=[nbformat.v4.new_markdown_cell('# V23 locally qualified candidate\n\n75% exactly reproduced V14 plus 25% LimiX-2 with the frozen training-only adapter and two fixed members. Local qualification is not a portal score. Upload the four original CSVs, use a free T4 and Run all. The inherited V14 section reconstructs its predictions from models; the new component also runs fresh inference. Expected predictions are used only after inference to verify reproduction. No automatic portal upload.'),
        nbformat.v4.new_code_cell("import os\nfrom pathlib import Path\nV23_OUTPUT=Path(os.environ.get('MLA_V23_OUTPUT_DIR','.')).resolve();V23_OUTPUT.mkdir(parents=True,exist_ok=True)\nos.environ['MLA_AUTOMATED_RUN']='1'\nos.environ['MLA_V14_OUTPUT_DIR']=str(V23_OUTPUT/'v14_reconstruction')\n")]
    for cell in baseline.cells[:-2]:
        cell=copy.deepcopy(cell)
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
        book.cells.append(cell)
    code="""import base64,zlib,json,hashlib,tempfile,importlib.metadata,io
import numpy as np
import pandas as pd
V23_WORK=Path(tempfile.mkdtemp(prefix='v23_fresh_',dir=V23_OUTPUT))
V23_BUNDLE=json.loads(zlib.decompress(base64.b64decode(PACKED)))
for name,encoded in V23_BUNDLE.items():
 relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
 path=V23_WORK/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(encoded))
V23_RUNTIME=RUNTIME
for name,version in V23_RUNTIME['versions'].items():assert importlib.metadata.version(name)==version,(name,version)
V23_AUDIT=AUDIT
for name,want in V23_AUDIT['inputs'].items():assert hashlib.sha256((DATA_DIR/name).read_bytes()).hexdigest()==want
V23_SOURCE=json.loads((V23_WORK/'research_sources/source_manifest.json').read_text())
for name,want in V23_SOURCE['files_sha256'].items():assert hashlib.sha256((V23_WORK/'research_sources/LimiX'/name).read_bytes()).hexdigest()==want
sys.path.insert(0,str(WORK))
import legacy_models as v23_legacy
v23_train,v23_test,v23_y=v23_legacy.load_data(DATA_DIR)
v23_features=v23_legacy.features(v23_train);v23_queries=v23_legacy.features(v23_test)
# Reproduce the declared feature-CSV round trip from freshly computed features.
for name,frame in [('feature_train.csv',v23_features),('feature_test.csv',v23_queries)]:
 raw=frame.to_csv(index=False).encode();assert hashlib.sha256(raw).hexdigest()==V23_AUDIT['files_sha256'][name]
v23_features=pd.read_csv(io.StringIO(v23_features.to_csv(index=False)))
v23_queries=pd.read_csv(io.StringIO(v23_queries.to_csv(index=False)))
sys.path.insert(0,str(V23_WORK))
import adapter as v23_adapter
v23_record=json.loads((V23_WORK/'research_sources/checkpoint_manifest.json').read_text())
v23_model_path=Path(os.environ.get('MLA_V23_MODEL_PATH',''))
if not v23_model_path.is_file():
 from huggingface_hub import hf_hub_download
 v23_model_path=Path(hf_hub_download(repo_id=v23_record['repo_id'],filename='LimiX-2.ckpt',revision=v23_record['revision'],token=False))
v23_hash=hashlib.sha256()
with v23_model_path.open('rb') as stream:
 for block in iter(lambda:stream.read(8*1024*1024),b''):v23_hash.update(block)
assert v23_hash.hexdigest()==V23_RUNTIME['checkpoint_sha256']
v23_model=v23_adapter.make_predictor(v23_model_path)
v23_state=v23_adapter.fit_state(v23_features,v23_y)
v23_component=v23_adapter.infer(v23_model,v23_state,v23_queries)
v23_adapter.release(v23_model)
v23_base_frame=pd.read_csv(OUTPUT_DIR/'submission.csv')
assert v23_base_frame['Registration Number'].equals(v23_test['Registration Number'])
v23_prediction=.75*v23_base_frame['Annual Turnover'].to_numpy(float)+.25*v23_component
v23_frame=pd.DataFrame({'Registration Number':v23_test['Registration Number'],'Annual Turnover':v23_prediction})
v23_legacy.ref.check_submission(v23_frame,v23_test)
v23_csv=V23_OUTPUT/'submission.csv';v23_frame.to_csv(v23_csv,index=False,float_format='%.12f')
# Verification witnesses are opened only after the fresh prediction and CSV exist.
v23_expected=np.frombuffer(base64.b64decode(WITNESS),dtype='<f8')
v23_diff=float(np.max(abs(v23_prediction-v23_expected)))
assert v23_diff<.01
v23_roundtrip=pd.read_csv(v23_csv);assert np.max(abs(v23_roundtrip['Annual Turnover']-v23_prediction))<.01
v23_checks={'status':'PASS','fresh_baseline_reproduced':checks['status']=='PASS','training_rows':3493,'query_rows':500,'maximum_final_difference_INR':v23_diff,'csv_sha256':hashlib.sha256(v23_csv.read_bytes()).hexdigest(),'portal_score':None,'portal_uploads':0}
(V23_OUTPUT/'V23_fresh_checks.json').write_text(json.dumps(v23_checks,indent=2)+'\\n')
print(v23_checks)
"""
    code=code.replace('PACKED',repr(packed)).replace('V23_RUNTIME=RUNTIME','V23_RUNTIME='+repr(runtime)).replace('V23_AUDIT=AUDIT','V23_AUDIT='+repr(audit)).replace('WITNESS',repr(witness))
    book.cells+=[nbformat.v4.new_markdown_cell('## Fresh LimiX inference and complete blend checks'),nbformat.v4.new_code_cell(code)]
    return book

def main():
    import adapter,nbformat
    from nbclient import NotebookClient
    decision=e.read(ROOT/'results/decision.json');assert decision['qualified'] and decision['weight']==.25
    assert e.sha(ROOT/'reference/scored_v14.ipynb')==e.read(ROOT/'history_audit.json')['reference_sha256']['ipynb']
    e.identities();e.require(1.3*(3*e.read(ROOT/'results/pilot.json')['full_seconds']+180))
    train,test,c,_=e.dataset();model=e.create();state=adapter.fit_state(train,c['y'])
    try:component=adapter.infer(model,state,test)
    finally:adapter.release(model)
    baseline=ROOT/'reference/scored_v14.csv';assert e.sha(baseline)==e.read(ROOT/'history_audit.json')['reference_sha256']['csv']
    expected=.75*pd.read_csv(baseline)['Annual Turnover'].to_numpy(float)+.25*component
    delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True)
    np.savez_compressed(delivery/'full_component_witness.npz',prediction=component,expected=expected)
    book=build_notebook(expected);path=delivery/('V23_CANDIDATE_'+STEM+'.ipynb');nbformat.write(book,path)
    fresh=delivery/'fresh_execution';fresh.mkdir(exist_ok=True)
    os.environ.update(MLA_DATA_DIR=str(ROOT/'data'),MLA_V23_OUTPUT_DIR=str(fresh),MLA_V23_MODEL_PATH=str(ROOT/'weights/LimiX-2.ckpt'),MLA_AUTOMATED_RUN='1')
    try:NotebookClient(book,timeout=max(1,int(e.read(ROOT/'authorization.json')['training_stop_epoch']-time.time())),kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:nbformat.write(book,path)
    checks=e.read(fresh/'V23_fresh_checks.json');assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
    assert all(cell.execution_count is not None for cell in book.cells if cell.cell_type=='code')
    csv=fresh/(STEM+'.csv');shutil.copyfile(csv,delivery/('V23_CANDIDATE_'+STEM+'.csv'))
    e.save(delivery/'qualified_export.json',{'status':'PASS','fresh_notebook_executed':True,'checks':checks,'notebook_sha256':e.sha(path),'csv_sha256':e.sha(csv),'portal_uploads':0,'portal_score':None})
    print('Qualified candidate fresh notebook/CSV verified.',flush=True)
if __name__=='__main__':main()
