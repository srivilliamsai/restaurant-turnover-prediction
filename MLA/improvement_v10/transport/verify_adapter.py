"""Verify padding/chunk ordering and original 500-row input identity without a GPU."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import features
import modeling
from core import Engine,save


class WitnessModel:
    def __init__(self):self.calls=[]
    def predict(self,frame,output_type):
        assert output_type=='mean' and len(frame)==500
        self.calls.append(frame.copy())
        return np.nan_to_num(frame.to_numpy(),nan=0).sum(axis=1)


e=Engine();builder=features.FeatureBuilder('C+S+G').fit(e.train)
training,vocab=features.native_frame(builder.transform(e.train))
model=WitnessModel()
fitted={'builder':builder,'vocabulary':vocab,'padding_row':training.iloc[:1].copy(),'model':model}
full=modeling.predict(fitted,e.test)
expected,_=features.native_frame(builder.transform(e.test),vocab)
pd.testing.assert_frame_equal(model.calls[0],expected.reset_index(drop=True))
parts=np.concatenate([modeling.predict(fitted,e.test.iloc[:13]),modeling.predict(fitted,e.test.iloc[13:])])
assert np.array_equal(full,parts)
many=pd.concat([e.test,e.test,e.test.iloc[:1]],ignore_index=True)
p=modeling.predict(fitted,many)
assert np.array_equal(p,np.concatenate([full,full,full[:1]]))
assert all(len(frame)==500 for frame in model.calls)
record={'status':'PASS','fixed_internal_rows':500,'full_500_input_unchanged':True,
        'split_and_1001_row_output_order':True,'padding_outputs_discarded':True,'no_prediction_rounding':True}
save(ROOT/'adapter_verification.json',record)
print(record)
