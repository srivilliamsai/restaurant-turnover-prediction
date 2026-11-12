"""Remote resumed-runtime verification; original experiment identity is retained."""
import importlib.metadata
import json
import sys
from pathlib import Path
root=Path('/content/mla_v10');sys.path.insert(0,str(root))
from core import Engine,PINNED,sha,save
import modeling
e=Engine(root/'data',root/'results')
pilot=json.loads((e.out/'pilot.json').read_text())
assert pilot['status']=='PASS' and pilot['sources']==e.sources and pilot['inputs']==e.inputs
assert {n:importlib.metadata.version(n) for n in PINNED}==PINNED
assert modeling.execution_contract()==pilot['execution_contract']
receipt=json.loads((e.out/'budget_amendment.json').read_text())
assert receipt['global_cap_hours']==24 and receipt['free_compute_only']
assert e.cache.state()['budget_amendment_sha256']==sha(e.out/'budget_amendment.json')
save(e.out/'resumed_runtime_checks.json',{'status':'PASS','inputs':e.inputs,'sources':e.sources,
     'execution_contract':modeling.execution_contract(),'pilot_reused_after_exact_identity_checks':True,
     'budget_amendment_sha256':sha(e.out/'budget_amendment.json')})
print('PASS: resumed T4 inputs, model sources, exact dependency/checkpoint contract and authorized budget epoch.',flush=True)
