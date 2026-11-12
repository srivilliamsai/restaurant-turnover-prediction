"""Preserve the portal-confirmed model, and promote only a supported candidate."""
from pathlib import Path
import json,hashlib,shutil
import nbformat
import pandas as pd
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;OUT=ROOT/'results'
STEM='submission'
summary=json.loads((OUT/'summary.json').read_text())
checks=json.loads((OUT/'submission_checks.json').read_text())
uncached=json.loads((OUT/'uncached_notebook_check.json').read_text())
execution=json.loads((OUT/'notebook_execution.json').read_text())
assert uncached['passed'] and execution['full_fit_reproduced']
# Independent test runner's detailed output is preserved; no unresolved failures.
assert checks['passed'] and not checks['failures'] and checks['scope']=='CSV, backups and executed notebook'
book=nbformat.read(ROOT/(STEM+'.ipynb'),as_version=4);nbformat.validate(book)
for cell in book.cells:
    if cell.cell_type=='code':
        assert cell.execution_count is not None
        assert not any(o.output_type=='error' for o in cell.outputs)
confirmed=json.loads((ROOT/'confirmed_portal_result.json').read_text())
for ext in ['csv','ipynb']:
    backup=ROOT/'rank3_confirmed_backup'/(STEM+'.'+ext)
    assert hashlib.sha256(backup.read_bytes()).hexdigest()==confirmed['files'][ext]['sha256']
frame=pd.read_csv(ROOT/(STEM+'.csv'))
assert frame.shape==(500,2) and frame.columns.tolist()==['Registration Number','Annual Turnover']
promoted=summary['supported_for_trial']
files={}
for ext in ['csv','ipynb']:
    source=ROOT/(STEM+'.'+ext)
    if promoted:
        destination=BASE/source.name
        # Permit only the confirmed v3 file or the exact already-promoted candidate.
        if destination.exists():
            previous_hash=hashlib.sha256(destination.read_bytes()).hexdigest()
            assert previous_hash in [confirmed['files'][ext]['sha256'],hashlib.sha256(source.read_bytes()).hexdigest()]
        shutil.copy2(source,destination)
    files[ext]={'path':str((BASE if promoted else ROOT)/source.name),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
manifest={'version':'v4','candidate_portal_rmse':None,'confirmed_v3_portal_rmse':confirmed['latest_portal_rmse'],
          'confirmed_rank':3,'leader_rmse_from_screenshot':confirmed['leader_rmse'],'files':files,
          'promoted_to_root':promoted,'summary':summary,'code_cells':execution['code_cells'],
          'validation_execution':execution['validation_execution'],'uncached_validation_checked':True,
          'backup':str(ROOT/'rank3_confirmed_backup')}
(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
