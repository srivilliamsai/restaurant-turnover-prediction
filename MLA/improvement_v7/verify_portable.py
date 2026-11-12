"""Check embedded Colab source portability and checkpoint rejection/recovery."""
import ast
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    book = json.loads((ROOT / 'V7_Research_Colab.ipynb').read_text())
    payload = {}
    for cell in book['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        compile(source, '<research notebook>', 'exec')
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ['SOURCE_FILES', 'PRESERVED_V5']:
                    payload[name] = json.loads(ast.literal_eval(node.value.args[0]))
    temporary = Path(tempfile.mkdtemp(prefix='mla_v7_portability_'))
    work = temporary / 'improvement_v7'
    work.mkdir()
    for name, source in payload['SOURCE_FILES'].items():
        (work / name).write_text(source)
    backup = work / 'scored_v5_backup'
    backup.mkdir()
    for name, value in payload['PRESERVED_V5'].items():
        (backup / name).write_bytes(base64.b64decode(value))
    # This folder deliberately has no v2/v3/v4/v5/v6 directories.
    checker = '''import json,sys
from pathlib import Path
import numpy as np
from workflow import Research, OUT
from prepare_resume import prepare
run=Research(Path(sys.argv[1]))
run.audit()
context=run.context('cache_behavior_test',extra={'purpose':'cache integrity only'})
calls=[]
def compute():
    calls.append(True)
    return np.array([1.,2.]), {'actual_computation':True}
p,_=run.cached('guard_valid',context,compute)
q,_=run.cached('guard_valid',context,lambda: (_ for _ in ()).throw(AssertionError('Should have reused verified checkpoint')))
assert len(calls)==1 and np.array_equal(p,q)
stale=False
try:run.cached('guard_valid',{**context,'extra':{'different':True}},compute)
except AssertionError:stale=True
assert stale
path=OUT/'cache/guard_valid.npz'
original=path.read_bytes()
path.write_bytes(original+b'changed')
corrupt=False
try:run.cached('guard_valid',context,compute)
except AssertionError:corrupt=True
assert corrupt
partial=OUT/'cache/guard_partial.npz'
np.savez_compressed(partial,prediction=np.array([999.]))
restored,_=run.cached('guard_partial',context,compute)
assert np.array_equal(restored,np.array([1.,2.]))
orphan=OUT/'cache/residual_outer_checkpoint.npz'
np.savez_compressed(orphan,prediction=np.array([999.]))
recovered=prepare(Path.cwd())
assert not orphan.exists() and len(recovered)==1 and Path(recovered[0]).exists()
report={'portable_audit_without_older_folders':True,'verified_cache_reused':True,
        'stale_context_rejected':stale,'corrupt_content_rejected':corrupt,
        'interrupted_write_without_metadata_recomputed':True,
        'orphan_residual_checkpoint_quarantined_before_resume':True,
        'note':'These are loader/checkpoint tests, not model accuracy tests.'}
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\\n')
print(json.dumps(report,indent=2))
'''
    subprocess.run([sys.executable, '-c', checker, str(ROOT.parent),
                    str(ROOT / 'results/portable_and_cache_checks.json')],
                   cwd=work, env=os.environ.copy(), check=True)
