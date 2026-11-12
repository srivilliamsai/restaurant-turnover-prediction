"""Retrain a complete validation assignment through embedded notebook code."""
from pathlib import Path
import json,time,io,contextlib
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
        if 'cache_value=os.environ.get' in cell.source:break
        exec(compile(cell.source,'setup_cell','exec'),space)
assert 'evaluate' in space and 'CONTRACT' in space
print('Notebook setup complete; retraining five folds for split 18421 without prediction caches.',flush=True)
start=time.time()
computed=space['evaluate'](space['train'],space['y'],18421,space['CONTRACT'])
expected=np.load(ROOT/'results/comparison_18421.npz')
errors={key:float(np.max(np.abs(computed[key]-expected[key]))) for key in computed}
for key in computed:assert np.allclose(computed[key],expected[key],rtol=1e-10,atol=.01),(key,errors[key])
result={'split_seed':18421,'folds':5,'cache_reused':False,'maximum_prediction_difference_INR':errors,'elapsed_seconds':time.time()-start,'passed':True}
(ROOT/'results/uncached_notebook_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
