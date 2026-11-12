"""Behavioral feature-isolation and cache tests, plus real-model inference checks."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from core import Cache, digest, save, validate_csv
import features


def feature_checks(e):
    checks = {}
    a,b=e.splits(42)[0]
    train=e.train.iloc[a];valid=e.train.iloc[b].iloc[:20].copy()
    for name in features.SPECS:
        builder=features.FeatureBuilder(name).fit(train)
        state=digest(builder.state());X=builder.transform(valid)
        changed=valid.copy();changed['Annual Turnover']=1e30;changed['Registration Number']=-9
        pd.testing.assert_frame_equal(X,builder.transform(changed))
        altered_train=train.copy();altered_train['Annual Turnover']=1e30;altered_train['Registration Number']=-8
        other=features.FeatureBuilder(name).fit(altered_train)
        assert digest(other.state())==state
        pd.testing.assert_frame_equal(X,other.transform(valid))
        perm=np.random.default_rng(91).permutation(len(valid))
        pd.testing.assert_frame_equal(builder.transform(valid.iloc[perm]),X.iloc[perm].reset_index(drop=True))
        separate=pd.concat([builder.transform(valid.iloc[:7]),builder.transform(valid.iloc[7:])],ignore_index=True)
        pd.testing.assert_frame_equal(X,separate)
        if features.SPECS[name]['prune']:
            changed=valid.copy();changed[features.REMOVED]=1e20
            pd.testing.assert_frame_equal(X,builder.transform(changed))
            changed=train.copy();changed[features.REMOVED]=-1e20
            other=features.FeatureBuilder(name).fit(changed)
            assert digest(other.state())==state
            pd.testing.assert_frame_equal(X,other.transform(valid))
        frame,vocabulary=features.native_frame(builder.transform(train))
        edge=valid.iloc[:3].copy();edge['City']='NEVER_SEEN_V10';edge['Cuisine']=pd.NA
        edge['Restaurant Theme']='NEVER_SEEN_THEME_V10';edge['Opening Day of Restaurant']='invalid'
        for col in features.SOCIAL+features.EVENTS+features.FACILITIES+['Resturant Tier']:
            edge[col]=np.nan
        transformed=builder.transform(edge)
        native,_=features.native_frame(transformed,vocabulary)
        assert native['City'].isna().all() and native['Restaurant Theme'].isna().all()
        if 'V10 Events count' in transformed:
            assert transformed['V10 Events count'].eq(0).all()
            assert transformed['V10 Events mean'].isna().all()
            assert transformed['V10 Facilities count'].isna().all()
        assert digest(builder.state())==state
        assert len(X.columns)==features.COUNTS[name]
        checks[name]={'features':len(X.columns),'target_and_id_exclusion':True,
                     'outer_validation_label_isolation':True,'batch_invariance':True,
                     'missing_and_unseen':True,'pruned_input_invariance':bool(features.SPECS[name]['prune'])}
    # Verify an unseen group's frequency and deviations use training-only fallbacks.
    builder=features.FeatureBuilder('C+S+G').fit(train)
    unknown=valid.iloc[:1].copy();unknown['City']='never_seen';unknown['Cuisine']='never_seen'
    x=builder.transform(unknown)
    for key in ['city','cuisine']:
        assert x[f'V10 Log {key} count'].iloc[0]==0
        for num,series in features.contexts(x)[1].items():
            expected=series.iloc[0]-builder.maps[key]['global'][num]
            assert np.isclose(x[f'V10 {num} relative {key}'].iloc[0],expected,equal_nan=True)
    with tempfile.TemporaryDirectory(dir=e.out) as tmp:
        cache=Cache(tmp);context={'inputs':'a','fold':[1,2]}
        p=cache.get('good',context,lambda:np.array([1.,2.]))
        assert np.array_equal(cache.get('good',context,lambda:(_ for _ in ()).throw(AssertionError())),p)
        for key,change in [('stale',None),('corrupt',None),('incomplete',None)]:
            if key!='incomplete':cache.get(key,context,lambda:p)
            if key=='corrupt':(cache.root/'corrupt.npz').write_bytes(b'corrupt')
            if key=='incomplete':np.savez_compressed(cache.root/'incomplete.npz',prediction=p)
            try:cache.get(key,{'changed':True} if key=='stale' else context,lambda:p)
            except (AssertionError,ValueError):pass
            else:raise AssertionError(f'{key} checkpoint accepted')
    checks['checkpoint_rejection']={'stale':True,'corrupt':True,'incomplete':True}
    checks['status']='PASS'
    save(e.out/'feature_verification.json',checks)
    return checks


def inference_checks(predict,raw,pruned=False):
    sample=raw.iloc[:12].copy();p=predict(sample)
    assert np.isfinite(p).all() and (p>=0).all()
    perm=np.random.default_rng(91).permutation(len(sample))
    assert np.max(np.abs(predict(sample.iloc[perm])-p[perm]))<.01
    changed=sample.copy();changed['Annual Turnover']=1e30;changed['Registration Number']=-7
    assert np.max(np.abs(predict(changed)-p))<.01
    split=np.concatenate([predict(sample.iloc[:5]),predict(sample.iloc[5:])])
    difference=float(np.max(np.abs(split-p)))
    assert difference<.01,f'Prediction batch dependence: INR {difference}'
    single=float(np.max(np.abs(np.asarray([predict(sample.iloc[i:i+1])[0] for i in range(2)])-p[:2])))
    assert single<.01,f'Individual prediction dependence: INR {single}'
    if pruned:
        changed=sample.copy();changed[features.REMOVED]=1e30
        assert np.max(np.abs(predict(changed)-p))<.01
    edge=sample.iloc[:3].copy();edge['City']='never_seen_v10';edge['Cuisine']=pd.NA
    edge['Restaurant Theme']='never_seen_v10';edge['Opening Day of Restaurant']='invalid'
    for c in features.SOCIAL+features.EVENTS+features.FACILITIES+['Resturant Tier']:edge[c]=np.nan
    q=predict(edge)
    assert np.isfinite(q).all() and (q>=0).all()
    return {'row_permutation':True,'target_and_id_invariance':True,'missing_unseen_events':True,
            'batch_difference_INR':difference,'individual_difference_INR':single,'pruned_invariance':pruned}


def csv_checks(e,p):
    frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p})
    validate_csv(frame,e.test)
    bads=[frame.iloc[:-1],pd.concat([frame,frame.iloc[:1]],ignore_index=True),frame.iloc[::-1].reset_index(drop=True),
          frame.iloc[:,::-1],frame.rename(columns={'Annual Turnover':'Turnover'}),frame.assign(extra=0)]
    for c,v in [('Registration Number','wrong'),('Registration Number',frame['Registration Number'].iloc[1]),
                ('Annual Turnover',np.nan),('Annual Turnover',np.inf),('Annual Turnover',-np.inf),('Annual Turnover',-1.)]:
        bad=frame.copy()
        if isinstance(v,str):bad[c]=bad[c].astype(object)
        bad.loc[0,c]=v;bads.append(bad)
    for bad in bads:
        try:validate_csv(bad,e.test)
        except (AssertionError,ValueError,TypeError):continue
        raise AssertionError('Malformed CSV accepted')
    return {'rows':500,'malformed_cases_rejected':len(bads)}
