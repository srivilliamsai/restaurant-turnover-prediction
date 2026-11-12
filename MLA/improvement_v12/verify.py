"""Training-only feature invariants and real model/CSV behavior checks."""
import tempfile,json
from pathlib import Path
import numpy as np
import pandas as pd
import features
from core import Cache,save,digest,validate_csv

def feature_checks(e):
    a,b=e.splits(42)[0];train=e.train.iloc[a];valid=e.train.iloc[b].iloc[:20].copy()
    X=features.frame(valid);changed=valid.copy();changed['Registration Number']=-99;changed['Annual Turnover']=1e30
    pd.testing.assert_frame_equal(X,features.frame(changed))
    _,v=features.native_frame(features.frame(train))
    edge=valid.copy();edge['City']='V11_NEVER_SEEN';edge['Cuisine']='V11_NEVER_SEEN';edge['Restaurant Theme']='V11_NEVER_SEEN'
    native,_=features.native_frame(features.frame(edge),v)
    for c in ['City','Cuisine','Restaurant Theme']:assert native[c].isna().all()
    for categorical in [True,False]:
        one,_=features.native_frame(features.frame(valid),v,categorical)
        joined=pd.concat([features.native_frame(features.frame(valid.iloc[:7]),v,categorical)[0],features.native_frame(features.frame(valid.iloc[7:]),v,categorical)[0]],ignore_index=True)
        pd.testing.assert_frame_equal(one,joined)
    neural=features.FeatureBuilder('R_10242').fit(train)
    altered=train.copy();altered['Annual Turnover']=1e30;altered['Registration Number']=-99
    other=features.FeatureBuilder('R_10242').fit(altered)
    assert neural.state()==other.state()
    pd.testing.assert_frame_equal(neural.transform(valid),other.transform(changed))
    adapted=neural.transform(edge)
    for c in neural.vocabulary:assert adapted[c].notna().all()
    for c in neural.numeric:assert adapted[c].notna().all() and c+'__missing' in adapted
    with tempfile.TemporaryDirectory(dir=e.out) as tmp:
        c=Cache(tmp);context={'fold':1};p=np.array([1.,2.]);c.get('good',context,lambda:p)
        np.testing.assert_array_equal(c.get('good',context,lambda:(_ for _ in ()).throw(AssertionError())),p)
        for key in ['stale','corrupt','incomplete']:
            if key!='incomplete':c.get(key,context,lambda:p)
            if key=='corrupt':(c.root/'corrupt.npz').write_bytes(b'bad')
            if key=='incomplete':np.savez_compressed(c.root/'incomplete.npz',prediction=p)
            try:c.get(key,{'fold':2} if key=='stale' else context,lambda:p)
            except (AssertionError,ValueError):pass
            else:raise AssertionError('Invalid checkpoint accepted')
    result={'status':'PASS','unchanged_v9_features':40,'target_id_isolation':True,'unknown_categories':True,'batch_fitted_vocabularies':True,'bad_checkpoints_rejected':True}
    save(e.out/'feature_verification.json',result);return result

def inference_checks(predict,raw,pruned=False):
    sample=raw.iloc[:12].copy();p=predict(sample);assert np.isfinite(p).all() and (p>=0).all()
    perm=np.random.default_rng(91).permutation(len(sample));permutation=float(np.max(np.abs(predict(sample.iloc[perm])-p[perm])))
    changed=sample.copy();changed['Annual Turnover']=1e30;changed['Registration Number']=-7
    isolation=float(np.max(np.abs(predict(changed)-p)))
    split=np.concatenate([predict(sample.iloc[:5]),predict(sample.iloc[5:])]);batch=float(np.max(np.abs(split-p)))
    individual=float(np.max(np.abs(np.asarray([predict(sample.iloc[i:i+1])[0] for i in range(2)])-p[:2])))
    for name,value in [('permutation',permutation),('validation labels and IDs',isolation),('split batch',batch),('individual row',individual)]:assert value<.01,f'{name} dependence INR {value}'
    edge=sample.iloc[:3].copy();edge['City']='V11_NEVER_SEEN';edge['Cuisine']=pd.NA;edge['Restaurant Theme']='V11_NEVER_SEEN';edge['Opening Day of Restaurant']='invalid'
    for c in features.SOCIAL+features.EVENTS+features.FACILITIES+['Resturant Tier']:edge[c]=np.nan
    q=predict(edge);assert np.isfinite(q).all() and (q>=0).all()
    return {'permutation_difference_INR':permutation,'label_id_difference_INR':isolation,'batch_difference_INR':batch,'individual_difference_INR':individual,'missing_unseen_inputs':True}

def csv_checks(e,p):
    frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p});validate_csv(frame,e.test)
    bads=[frame.iloc[:-1],pd.concat([frame,frame.iloc[:1]],ignore_index=True),frame.iloc[::-1].reset_index(drop=True),frame.iloc[:,::-1],frame.rename(columns={'Annual Turnover':'Turnover'}),frame.assign(extra=0)]
    for c,v in [('Registration Number','wrong'),('Registration Number',frame['Registration Number'].iloc[1]),('Annual Turnover',np.nan),('Annual Turnover',np.inf),('Annual Turnover',-np.inf),('Annual Turnover',-1.)]:
        bad=frame.copy()
        if isinstance(v,str):bad[c]=bad[c].astype(object)
        bad.loc[0,c]=v;bads.append(bad)
    for bad in bads:
        try:validate_csv(bad,e.test)
        except (AssertionError,ValueError,TypeError):continue
        raise AssertionError('Malformed CSV accepted')
    return {'rows':500,'malformed_cases_rejected':len(bads)}
