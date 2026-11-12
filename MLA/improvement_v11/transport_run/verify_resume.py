import sys,json
from pathlib import Path
root=Path('/content/mla_v11');sys.path.insert(0,str(root))
from core import Engine,sha,save
import modeling,workflow
e=Engine(root/'data',root/'results');p=json.loads((e.out/'pilot.json').read_text())
assert p['status'] in ['PASS','NO_FEASIBLE_FAMILY']
assert p['sources']==e.sources and p['inputs']==e.inputs
assert p['model_contract_sha256']==sha(root/'resolved_models.json')
assert p['model_contract']['runtime']==modeling.runtime()
assert callable(workflow.diagnostics) and callable(workflow.uncertainty)
for family in {name[0] for name in p['screen_settings']}:
 assert p['families'][family]['status']=='PASS'
 modeling.local_checkpoint(p['model_contract']['models'][family]['checkpoint'])
save(e.out/'resume_verification.json',{'status':'PASS','pilot_reused_without_retuning':True,'sources':e.sources,'inputs':e.inputs,'model_contract_sha256':p['model_contract_sha256'],'entrypoint':'import workflow; workflow.main()'})
print('Frozen pilot, runtime, checkpoint and fully loaded reporting functions verified; no model refit for resume.')
