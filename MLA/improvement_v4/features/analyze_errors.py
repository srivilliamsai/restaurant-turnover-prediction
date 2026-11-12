"""Diagnostics on the existing development OOF arrays, no new model fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from screen_features import feature_frame, RATINGS, ROOT, OUT, rmse

train=pd.read_csv('~/Downloads/Train_dataset.csv')
test=pd.read_csv('~/Downloads/Test_dataset.csv')
y=train['Annual Turnover'].to_numpy(float)
v3=.5*np.load(ROOT/'improvement_v3/results/v2_42.npz')['oof']+.5*np.load(ROOT/'improvement_v3/results/bag_subsample_h40_42.npz')['oof']
cat=np.load(OUT/'categorical_ratings_42.npz')['oof']
candidate=.5*v3+.5*cat
old_sq=(y-v3)**2
new_sq=(y-candidate)**2
change=candidate-v3

folds=[]
for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=42).split(train),1):
    folds.append(dict(fold=fold,n=len(va),v3_rmse=rmse(y[va],v3[va]),candidate_rmse=rmse(y[va],candidate[va]),gain=rmse(y[va],v3[va])-rmse(y[va],candidate[va]),tail_gt100m=int((y[va]>1e8).sum())))
pd.DataFrame(folds).to_csv(OUT/'fold_errors.csv',index=False)

segments=[]
for label,mask in [('all',np.ones(len(y),bool)),('<=100m',y<=1e8),('>100m',y>1e8),('>200m',y>2e8)]:
    segments.append(dict(segment=label,n=int(mask.sum()),v3_rmse=rmse(y[mask],v3[mask]),candidate_rmse=rmse(y[mask],candidate[mask]),gain=rmse(y[mask],v3[mask])-rmse(y[mask],candidate[mask]),v3_squared_error_share=float(old_sq[mask].sum()/old_sq.sum()),candidate_squared_error_share=float(new_sq[mask].sum()/new_sq.sum()),total_squared_error_reduction_share=float((old_sq[mask]-new_sq[mask]).sum()/(old_sq-new_sq).sum())))
pd.DataFrame(segments).to_csv(OUT/'tail_errors.csv',index=False)
deciles=pd.qcut(pd.Series(y),10,duplicates='drop')
decile_stats=[]
for interval in deciles.cat.categories:
    mask=(deciles==interval).to_numpy()
    decile_stats.append(dict(bin=str(interval),n=int(mask.sum()),min_target=float(y[mask].min()),max_target=float(y[mask].max()),v3_rmse=rmse(y[mask],v3[mask]),candidate_rmse=rmse(y[mask],candidate[mask]),gain=rmse(y[mask],v3[mask])-rmse(y[mask],candidate[mask]),candidate_squared_error_share=float(new_sq[mask].sum()/new_sq.sum())))
pd.DataFrame(decile_stats).to_csv(OUT/'decile_errors.csv',index=False)

# Paired restaurant resampling, conditional on these already selected/fitted OOF predictions.
rng=np.random.default_rng(4917)
gains=[]
for batch in range(50):
    ix=rng.integers(0,len(y),size=(100,len(y)))
    gains.extend((np.sqrt(old_sq[ix].mean(axis=1))-np.sqrt(new_sq[ix].mean(axis=1))).tolist())
bootstrap=dict(repetitions=5000,seed=4917,ci95=np.percentile(gains,[2.5,97.5]).tolist(),fraction_positive=float(np.mean(np.asarray(gains)>0)),interpretation='Conditional paired case-bootstrap; model/feature selection optimism and retraining variability are not represented. Development-only, not independent confirmation.')

trframe=feature_frame(train,'categorical_ratings')
teframe=feature_frame(test,'categorical_ratings')
coverage=[]
for c in RATINGS:
    trset=set(trframe[c]);unknown=set(teframe[c])-trset
    nonmissing=pd.concat([train[c],test[c]]).dropna().to_numpy(float)
    category_unique=len({f'{v:g}' for v in nonmissing})
    numeric_unique=len(set(nonmissing))
    unseenval=0
    for tr,va in KFold(5,shuffle=True,random_state=42).split(train):
        unseenval+=int((~trframe[c].iloc[va].isin(set(trframe[c].iloc[tr]))).sum())
    coverage.append(dict(feature=c,train_categories=len(trset),test_categories=int(teframe[c].nunique()),unknown_test_categories=sorted(unknown),test_rows_unknown=int((~teframe[c].isin(trset)).sum()),unseen_validation_rows=unseenval,non_integer_values=int(np.sum(nonmissing!=np.floor(nonmissing))),numeric_to_string_collisions=numeric_unique-category_unique))
pd.DataFrame(coverage).to_csv(OUT/'category_coverage.csv',index=False)

summary=dict(v3_rmse=rmse(y,v3),candidate_rmse=rmse(y,candidate),gain=rmse(y,v3)-rmse(y,candidate),bootstrap=bootstrap,prediction_change=dict(mean=float(change.mean()),mean_absolute=float(np.abs(change).mean()),median_absolute=float(np.median(np.abs(change))),p95_absolute=float(np.quantile(np.abs(change),.95)),maximum_absolute=float(np.abs(change).max()),fraction_increased=float(np.mean(change>0))),foldwise=folds,segments=segments,coverage=coverage)
(OUT/'error_analysis.json').write_text(json.dumps(summary,indent=2))

lines=['# Feature representation error analysis','',
       'Development predictions only: shuffled 5-fold seed42, all original turnover labels. Candidate = 50% existing v3 + 50% two-seed categorical-ratings CatBoost. No models were refitted and no new validation splits were used.','',
       f'Overall RMSE: v3 **{rmse(y,v3):,.2f}**, candidate **{rmse(y,candidate):,.2f}**, reduction **{rmse(y,v3)-rmse(y,candidate):,.2f}**.','',
       '## Fold consistency','', '| Fold | v3 RMSE | Candidate RMSE | Reduction | >100m rows |','|---|---:|---:|---:|---:|']
for row in folds: lines.append(f"| {row['fold']} | {row['v3_rmse']:,.2f} | {row['candidate_rmse']:,.2f} | {row['gain']:,.2f} | {row['tail_gt100m']} |")
lines+=['','## Tail sensitivity','', '| Segment | Rows | Reduction in RMSE | Candidate squared-error share | Share of total squared-error reduction |','|---|---:|---:|---:|---:|']
for row in segments: lines.append(f"| {row['segment']} | {row['n']} | {row['gain']:,.2f} | {100*row['candidate_squared_error_share']:.2f}% | {100*row['total_squared_error_reduction_share']:.2f}% |")
lines+=['','The >200m segment overlaps >100m. Negative reduction indicates deterioration. Target groups are diagnostic only; none are excluded from selection.','','## Target deciles','', '| Target interval | Rows | Reduction in RMSE | Candidate error share |','|---|---:|---:|---:|']
for row in decile_stats: lines.append(f"| {row['bin']} | {row['n']} | {row['gain']:,.2f} | {100*row['candidate_squared_error_share']:.2f}% |")
lo,hi=bootstrap['ci95']
lines+=['','## Uncertainty and prediction changes','',f'Conditional paired case-bootstrap: 5,000 resamples, RMSE-reduction 95% interval **[{lo:,.2f}, {hi:,.2f}]**; positive fraction **{100*bootstrap["fraction_positive"]:.1f}%**. These are fixed, selected development OOF predictions. The interval omits feature-selection optimism, model retraining variation, and test distribution shift; it is not a guarantee of generalization.','',f'OOF prediction changes: mean {change.mean():,.2f}; mean absolute {np.abs(change).mean():,.2f}; median absolute {np.median(np.abs(change)):,.2f}; 95th percentile absolute {np.quantile(np.abs(change),.95):,.2f}; maximum absolute {np.abs(change).max():,.2f}.','', '## Categorical representation audit','', '| Feature | Train categories | Test categories | Unseen test categories (rows) | Unseen dev validation rows |','|---|---:|---:|---|---:|']
for row in coverage: lines.append(f"| {row['feature']} | {row['train_categories']} | {row['test_categories']} | {', '.join(row['unknown_test_categories']) or 'None'} ({row['test_rows_unknown']}) | {row['unseen_validation_rows']} |")
lines+=['','Formatting uses `Missing` for nulls and `f\'{v:g}\'` otherwise, so integer/float representations of the same observed rating map to the same category. The inspected train and test rating values have no numeric-to-string collisions. Social popularity remains numeric. Unseen categories are valid CatBoost inputs and use learned fallback statistics; they do not cause invalid predictions, but performance for unseen values is uncertain.','', 'The conversion treats distance/order between rating values as something to learn through categorical statistics. Therefore it cannot smoothly extrapolate a rating trend beyond training support. The proposed blend retains 50% of v3, which contains numeric rating models. Independent checks are required before retaining this candidate.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
