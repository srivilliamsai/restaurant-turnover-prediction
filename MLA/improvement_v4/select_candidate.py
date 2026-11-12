from pipeline import *
from screen_parameters import COMMON,TRIALS
train,test=load_data();y=train['Annual Turnover'].to_numpy(float)
base=.5*np.load(BASE/'improvement_v3/results/v2_42.npz')['oof']+.5*np.load(BASE/'improvement_v3/results/bag_subsample_h40_42.npz')['oof']
specs={};preds={};families={}
for name,override in TRIALS.items():
    key='parameter_'+name;specs[key]={'variant':'base','parameters':{**COMMON,**override}}
    preds[key]=np.load(OUT/f'{name}_42.npz')['oof'];families[key]='parameter'
for name in json.loads((ROOT/'features/specifications.json').read_text())['variants']:
    key='feature_'+name;specs[key]={'variant':name,'parameters':COMMON.copy()}
    preds[key]=np.load(ROOT/'features'/f'{name}_42.npz')['oof'];families[key]='feature'
for value,label in [(.25,'0p25'),(.5,'0p5')]:
    key='support_'+label
    specs[key]={'variant':'base','parameters':COMMON.copy(),'support_columns':['Facebook Popularity Quotient','Instagram Popularity Quotient','Resturant Tier','Ambience'],'missing_weight':value}
    preds[key]=np.load(ROOT/'audit'/f'support_weight_{label}_42.npz')['oof'];families[key]='support'
# Bounded blend grid and two predetermined family-combination rules.
mixtures={k:{k:1.} for k in specs}
best={family:min((k for k in specs if families[k]==family),key=lambda k:rmse(y,.5*base+.5*preds[k])) for family in ['parameter','feature','support']}
mixtures['best_parameter_feature']={best['parameter']:.5,best['feature']:.5}
mixtures['best_each_family']={k:1/3 for k in best.values()}
rows=[]
for name,weights in mixtures.items():
    p=sum(weights[k]*preds[k] for k in weights)
    for w in [.25,.5,.75]:
        score=rmse(y,(1-w)*base+w*p)
        rows.append({'name':name,'new_weight':w,'rmse':score,'gain_vs_v3':rmse(y,base)-score})
table=pd.DataFrame(rows).sort_values('rmse');table.to_csv(OUT/'selection.csv',index=False)
winner=table.iloc[0];w=float(winner['new_weight']);components={}
for name,share in mixtures[winner['name']].items():components[name]={**specs[name],'weight':share*w}
v2=json.loads((BASE/'improvement_v2/results/locked_candidate.json').read_text())
v3=json.loads((BASE/'improvement_v3/results/candidate_lock.json').read_text())
contract={'version':'v4','v2_configs':v2['configs'],'v2_weights':json.loads((BASE/'improvement_v2/results/final_execution.json').read_text())['weights'],
 'v3_parameters':v3['specs']['bag_subsample_h40'],'v3_seeds':v3['model_seeds'],'v3_weight':1-w,
 'seeds':[1042,2042],'components':components,'development':{'baseline_rmse':rmse(y,base),'candidate_rmse':float(winner['rmse']),'settings_evaluated':len(specs),'blends_evaluated':len(rows),'selection_split':42},
 'validation_plan':{'new_splits':[1701,1711],'previous_splits':[71,2026,303,909],'stress':['city'],'warning':'Resplits of reused training labels, not an external test; no retuning on these checks.'}}
assert np.isclose(contract['v3_weight']+sum(c['weight'] for c in components.values()),1)
(OUT/'candidate_lock.json').write_text(json.dumps(contract,indent=2)+'\n')
print(table.head(8).to_string(index=False));print('LOCKED',json.dumps(components,indent=2))
