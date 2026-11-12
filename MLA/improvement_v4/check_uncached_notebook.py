"""Run the notebook's default validation implementation for a complete split."""
from pathlib import Path
import ast,json,time,io,contextlib
import numpy as np
import nbformat
import matplotlib
matplotlib.use('Agg')
ROOT=Path(__file__).resolve().parent
book=nbformat.read(ROOT/'submission.ipynb',as_version=4)
space={'__name__':'__main__'};setup_output=io.StringIO()
with contextlib.redirect_stdout(setup_output):
    for cell in book.cells:
        if cell.cell_type!='code':continue
        if 'def compute_validation(' in cell.source:
            nodes=[n for n in ast.parse(cell.source).body if isinstance(n,ast.FunctionDef)]
            for node in nodes:
                exec(compile(ast.Module(body=[node],type_ignores=[]),'validation_functions','exec'),space)
            break
        exec(compile(cell.source,'setup_cell','exec'),space)
assert 'compute_validation' in space
print('Notebook setup complete; retraining all five folds for split 1701 without any prediction cache.',flush=True)
start=time.time()
computed=space['compute_validation'](space['train'],space['y'],1701,space['CONTRACT'],use_prior_cache=False)
expected=np.load(ROOT/'results/comparison_1701.npz')
errors={key:float(np.max(np.abs(computed[key]-expected[key]))) for key in computed}
for key in computed:assert np.allclose(computed[key],expected[key],rtol=1e-10,atol=.01),(key,errors[key])
result={'split_seed':1701,'folds':5,'cache_reused':False,'maximum_prediction_difference_INR':errors,'elapsed_seconds':time.time()-start,'passed':True}
(ROOT/'results/uncached_notebook_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
