"""Read-only environment inspection; no competition data is transferred."""
import json
import colab_control as c
from colab_cli.common import state
from colab_cli.contents import ContentsClient
code = """import importlib.metadata as m,inspect,json,hashlib
from pathlib import Path

versions={k:m.version(k) for k in ['numpy','pandas','scikit-learn','scipy','catboost','interpret-core','tabpfn','tabicl','numba','synthefy','einops','kditransform','pyvinecopulib','skrub','pyarrow']}
record={'versions':versions}
try:
    from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
    record['ft_signature']=str(inspect.signature(FinetunedTabPFNRegressor))
except Exception as error:
    import traceback
    record['import_error']=traceback.format_exc()
Path('/content/mla_v17/environment_review.json').write_text(json.dumps(record,indent=2))
"""
c.execute(code,120)
record=json.loads(c.read(ContentsClient(state.get_session(c.SESSION)),c.REMOTE+'/environment_review.json'))
(c.ROOT/'environment_review.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
