from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
train = pd.read_csv('~/Downloads/Train_dataset.csv').rename(columns={'Endorsed By':'Endoresed By'})
test = pd.read_csv('~/Downloads/Test_dataset.csv')
y = train['Annual Turnover'].to_numpy() / 1e6
nb = json.loads((ROOT.parent / 'submission.ipynb').read_text())
ns = dict(pd=pd, np=np, X_raw=train.drop(columns=['Annual Turnover','Registration Number']), X_test_raw=test.drop(columns='Registration Number'))
exec(''.join(nb['cells'][9]['source']), ns)
X, T, cats = ns['X'], ns['X_test'], ns['cat_cols']
print('Raw dates parsing:', pd.to_datetime(train['Opening Day of Restaurant'], format='%d-%m-%Y', errors='coerce').isna().sum())
print('Exact duplicate features:', X.duplicated().sum(), 'across train/test:', pd.concat([X,T]).duplicated().sum()-X.duplicated().sum()-T.duplicated().sum())
for threshold in [60,80,100,150,200]:
    mask=y>threshold
    print('Tail above',threshold,'million:',int(mask.sum()),'share of baseline squared error:',float(np.square(y-y.mean())[mask].sum()/np.square(y-y.mean()).sum()))
print('\nTarget and ratings by training row quarter:')
print(train.assign(part=pd.qcut(np.arange(len(train)),4)).groupby('part',observed=True)[['Annual Turnover','Service','Comfortablility']].agg(['mean','median']).to_string())
print('\nChanged categorical proportions:')
for c in cats:
    a=X[c].value_counts(normalize=True); b=T[c].value_counts(normalize=True)
    dif=(b-a).fillna(b).fillna(-a).sort_values(key=abs,ascending=False)
    print(c, dif.head(4).round(3).to_dict())
both=pd.concat([X,T],ignore_index=True)
domain=np.r_[np.zeros(len(X)),np.ones(len(T))]
pred=np.zeros(len(both)); importance=[]
for tr,va in StratifiedKFold(5,shuffle=True,random_state=809).split(both,domain):
    model=CatBoostClassifier(iterations=400,depth=4,learning_rate=.04,verbose=False,thread_count=4,allow_writing_files=False,random_seed=809)
    model.fit(both.iloc[tr],domain[tr],cat_features=cats)
    pred[va]=model.predict_proba(both.iloc[va])[:,1]
    importance.append(model.feature_importances_)
print('\nDomain classifier OOF AUC:',roc_auc_score(domain,pred))
imp=pd.Series(np.mean(importance,axis=0),index=both.columns).sort_values(ascending=False)
print(imp.head(15).to_string())
pd.DataFrame({'train_row':np.arange(len(X)),'prob_test_domain':pred[:len(X)]}).to_csv(ROOT/'results/domain_predictions.csv',index=False)
print('\nLargest numeric distribution differences:')
num=X.select_dtypes('number').columns
print(((T[num].mean()-X[num].mean())/X[num].std()).sort_values(key=abs,ascending=False).head(12).round(3).to_string())
