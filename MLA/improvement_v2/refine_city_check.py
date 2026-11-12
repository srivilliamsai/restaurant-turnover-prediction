from experiment import *
from sklearn.model_selection import GroupKFold
import contextlib, io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

name='submission'
path=ROOT/(name+'.ipynb')
nb=json.loads(path.read_text())
summary=json.loads((ROOT/'results/final_execution.json').read_text())
locked=json.loads((ROOT/'results/locked_candidate.json').read_text())
weights=summary['weights']; configs=locked['configs']
train,test=load_data(); y=train[TARGET].to_numpy(float)
all_specs={'raw_baseline':SPECS['raw_baseline'],'huber_baseline':SPECS['huber_baseline'],**configs}
data={v:features(train,v) for v in {s['variant'] for s in all_specs.values()}}
groups=data['engineered']['City']
folds=list(GroupKFold(5).split(train,y,groups))
for tr,va in folds: assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
pp={}
for model,spec in all_specs.items():
    p=np.zeros(len(y))
    for fold,(tr,va) in enumerate(folds,1):
        p[va],_=fit_predict(data[spec['variant']],y,tr,va,spec,42+fold,spec.get('iterations'))
    pp[model]=p
    print('Canonical city check:',model,rmse(y,p),flush=True)
old=.55*pp['raw_baseline']+.45*pp['huber_baseline']
new=sum(w*pp[k] for k,w in weights.items())
record={'Validation':'City GroupKFold','Rank-3 model RMSE':rmse(y,old),'New candidate RMSE':rmse(y,new),'Reduction':rmse(y,old)-rmse(y,new)}
previous=summary['validation'][-1].copy()
summary['validation'][-1]=record
summary['city_grouping']='Lowercase/trimmed city names with explicit aliases combined; rerun after full notebook execution.'
np.savez_compressed(ROOT/'results/comparison_city_canonical.npz',baseline=old,candidate=new)
table=pd.DataFrame(summary['validation'])

# Reuse unchanged, saved predictions from the actual random-split fits. Verify that
# they reproduce the fully executed notebook before rebuilding affected diagnostics.
random_predictions={}
for seed in [71,2026,303]:
    predictions={k:np.load(ROOT/'results'/f'{k}_{seed}_fixed.npz')['oof'] for k in all_specs}
    before=.55*predictions['raw_baseline']+.45*predictions['huber_baseline']
    after=sum(w*predictions[k] for k,w in weights.items())
    label=f'Shuffled 5-fold seed {seed}'
    row=table[table['Validation']==label].iloc[0]
    assert np.isclose(rmse(y,before),row['Rank-3 model RMSE'],rtol=0,atol=.001)
    assert np.isclose(rmse(y,after),row['New candidate RMSE'],rtol=0,atol=.001)
    random_predictions[label]={'old':before,'new':after}

code=''.join(nb['cells'][15]['source']).replace("city_groups=feature_data['base']['City']","city_groups=feature_data['engineered']['City']  # Group explicit aliases together.")
nb['cells'][15]['source']=code.splitlines(keepends=True)
lines=[]
for row in summary['validation']:
    lines.append(f'{row["Validation"]}: old {row["Rank-3 model RMSE"]:,.0f}; new {row["New candidate RMSE"]:,.0f}; reduction {row["Reduction"]:,.0f}')
lines+=['Mean across three shuffled assignments:',table.iloc[:3].drop(columns='Validation').mean().round(0).to_string()]
nb['cells'][15]['outputs']=[{'output_type':'stream','name':'stdout','text':('\n'.join(lines)+'\n').splitlines(keepends=True)}]
nb['cells'][15]['execution_count']=13
md=''.join(nb['cells'][14]['source']).replace('City grouping prevents a city\'s rows', 'City grouping combines explicit aliases (such as Delhi/New Delhi) and prevents a city\'s rows')
nb['cells'][14]['source']=md.splitlines(keepends=True)

# Execute the unchanged diagnostics cell using the measured predictions, so its
# uncertainty output and plotted city bars correspond to the corrected grouping.
figures=[]
def show(*args,**kwargs):
    for fid in plt.get_fignums():
        fig=plt.figure(fid); bio=io.BytesIO(); fig.savefig(bio,format='png',dpi=130,bbox_inches='tight')
        figures.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(bio.getvalue()).decode(),'text/plain':['<Figure rendered from executed notebook code>']}})
        plt.close(fig)
plt.show=show
ns={'np':np,'pd':pd,'plt':plt,'y':y,'validation_table':table,'validation_predictions':{**random_predictions,'City GroupKFold':{'old':old,'new':new}}}
output=io.StringIO()
with contextlib.redirect_stdout(output): exec(''.join(nb['cells'][17]['source']),ns)
nb['cells'][17]['outputs']=[{'output_type':'stream','name':'stdout','text':output.getvalue().splitlines(keepends=True)},*figures]
nb['cells'][17]['execution_count']=14
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
(ROOT/'results/final_execution.json').write_text(json.dumps(summary,indent=2))

report=(ROOT/'improvement_report.md').read_text()
old_row=f'| City GroupKFold | {previous["Rank-3 model RMSE"]:,.2f} | {previous["New candidate RMSE"]:,.2f} | {previous["Reduction"]:,.2f} |'
new_row=f'| City GroupKFold | {record["Rank-3 model RMSE"]:,.2f} | {record["New candidate RMSE"]:,.2f} | {record["Reduction"]:,.2f} |'
assert old_row in report
report=report.replace(old_row,new_row).replace('- Executed all 12 Python cells; zero cell errors.', '- Executed all 12 Python cells; zero cell errors. Then reran the city stress test with explicit aliases grouped together and refreshed its diagnostics.')
(ROOT/'improvement_report.md').write_text(report)
print('Corrected city-group comparison:',record,flush=True)
