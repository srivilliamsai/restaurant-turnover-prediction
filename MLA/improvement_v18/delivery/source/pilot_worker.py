"""Bounded actual-fold engineering subprocess, without development scoring."""
import argparse
import json
import pickle
import tempfile
import time
import traceback
from pathlib import Path
import numpy as np
import modeling
import verify
from core import Engine, ROOT, save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('name')
    args=parser.parse_args()
    e=Engine(ROOT/'data' if (ROOT/'data').exists() else None)
    a,b=e.splits(42)[0]
    name=args.name
    started=time.monotonic()
    fitted=repeated=full=None
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        fitted=modeling.fit(e.train.iloc[a],e.y[a],name,fold=0,scheme=42)
        p=modeling.predict(fitted,e.train.iloc[b])
        fold_seconds=time.monotonic()-started
        metadata=fitted['target_metadata']
        checks=verify.inference_checks(lambda raw:modeling.predict(fitted,raw),e.train.iloc[b],e.y[b])
        # Persist exactly the same fitted state. Prediction must not refit on queries.
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fitted.pkl'
            path.write_bytes(pickle.dumps(fitted,protocol=5))
            reloaded=pickle.loads(path.read_bytes())
            difference=float(np.max(np.abs(p-modeling.predict(reloaded,e.train.iloc[b]))))
            assert difference<.01,f'Saved/reloaded difference INR {difference}'
            modeling.release(reloaded)
        modeling.release(fitted);fitted=None
        repeated=modeling.fit(e.train.iloc[a],e.y[a],name,fold=0,scheme=42)
        other=modeling.predict(repeated,e.train.iloc[b])
        fresh=float(np.max(np.abs(p-other)))
        assert fresh<.01,f'Fresh repeated fit difference INR {fresh}'
        modeling.release(repeated);repeated=None
        full_started=time.monotonic()
        full=modeling.fit(e.train,e.y,name)
        full_output=modeling.predict(full,e.test)
        assert full_output.shape==(500,) and np.isfinite(full_output).all()
        full_seconds=time.monotonic()-full_started
        # No RMSE or eligible blend is calculated in this subprocess.
        record={'status':'PASS','fold_seconds':fold_seconds,'full_fit_predict_seconds':full_seconds,
            'behavior':checks,'fresh_repeat_difference_INR':fresh,'save_reload_difference_INR':difference,
            'target_metadata':metadata,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'training_rows':len(a),'validation_rows':len(b),'full_training_rows':len(e.y),
            'selection_uses_validation_rmse':False,'engineering_seconds':time.monotonic()-started}
        if name.startswith('FT'):
            def maximum_seconds(meta,query):
                return meta['setup_seconds']+meta['selection_seconds']*30/max(meta['inner_epochs_run'],1)+meta['refit_seconds']*30/max(meta['selected_epochs'],1)+query
            record['fold_seconds']=maximum_seconds(metadata,metadata.get('last_query_seconds',0.))
            record['full_fit_predict_seconds']=maximum_seconds(full['target_metadata'],full['target_metadata'].get('last_query_seconds',0.))
            record['forecast_basis']='Measured fixed setup, maximum thirty selection epochs, thirty refit epochs, full validation and final queries; thirty-percent contingency applied by admission.'
    except BaseException as error:
        record={'status':'engineering_excluded','error_type':type(error).__name__,
            'reason':str(error)[:1500],'traceback':traceback.format_exc(),
            'engineering_seconds':time.monotonic()-started,'accuracy_tested':False}
    finally:
        modeling.release(fitted);modeling.release(repeated);modeling.release(full)
    save(e.out/'pilot_checkpoints'/f'{name}.json',{'result':record,'result_sha256':__import__('core').digest(record)})
    print(json.dumps({'setting':name,**record}),flush=True)


if __name__=='__main__':
    main()
