"""Exercise the notebook's fresh Colab validation path on a complete five-fold run."""
from pathlib import Path
import ast,json,time
import numpy as np
import nbformat
import matplotlib
matplotlib.use('Agg')
ROOT=Path(__file__).resolve().parent
book=nbformat.read(ROOT/'submission.ipynb',as_version=4)
space={'__name__':'__main__'}
for c in book.cells:
    if c.cell_type!='code':continue
    if 'def get_splits(scheme):' in c.source:
        nodes=[n for n in ast.parse(c.source).body if isinstance(n,ast.FunctionDef)]
        for node in nodes:exec(compile(ast.Module(body=[node],type_ignores=[]),'validation_functions','exec'),space)
        break
    if 'fig,ax=plt.subplots' in c.source:continue
    exec(compile(c.source,'setup_cell','exec'),space)
start=time.time();computed=space['compute_validation'](909)
expected=np.load(ROOT/'results/comparison_909.npz')
errors={key:float(np.max(np.abs(computed[key]-expected[key]))) for key in computed}
for key in computed:assert np.allclose(computed[key],expected[key],rtol=1e-10,atol=.01),(key,errors[key])
result={'split_seed':909,'folds':5,'cache_reused':False,'maximum_prediction_difference_INR':errors,'elapsed_seconds':time.time()-start,'passed':True}
(ROOT/'results/uncached_notebook_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
