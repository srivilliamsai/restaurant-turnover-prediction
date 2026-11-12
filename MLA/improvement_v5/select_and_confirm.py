from pipeline import *
from new_models import SPECS,run_oof
import argparse,time

def baseline(seed):
    if seed==42:return .5*np.load(BASE/'improvement_v3/results/v2_42.npz')['oof']+.5*np.load(BASE/'improvement_v3/results/bag_subsample_h40_42.npz')['oof']
    return np.load(BASE/'improvement_v3/results'/f'comparison_{seed}.npz')['candidate']

def select():
    train,test=load_data();y=train['Annual Turnover'].to_numpy(float)
    table=pd.read_csv(OUT/'screen_42.csv');assert len(table)==len(SPECS)
    table['best_blend']=table[['blend25','blend50']].min(axis=1)
    names=table.sort_values('best_blend').head(6)['name'].tolist()
    (OUT/'shortlist.json').write_text(json.dumps(names,indent=2))
    for name in names:
        p=run_oof(train,y,name,71);print('Second development',name,rmse(y,p),flush=True)
    bases={s:baseline(s) for s in [42,71]}
    individual={n:{s:np.load(OUT/f'{n}_{s}.npz')['oof'] for s in [42,71]} for n in names}
    rank=sorted(names,key=lambda n:np.mean([rmse(y,.5*bases[s]+.5*individual[n][s]) for s in bases]))
    combos={n:{n:1.} for n in names}
    combos['top2']={n:.5 for n in rank[:2]};combos['top3']={n:1/3 for n in rank[:3]}
    rows=[]
    for name,components in combos.items():
        for w in [.25,.5,.75]:
            scores={s:rmse(y,(1-w)*bases[s]+w*sum(c*individual[n][s] for n,c in components.items())) for s in bases}
            gains={s:rmse(y,bases[s])-scores[s] for s in bases}
            rows.append({'name':name,'new_weight':w,'rmse42':scores[42],'rmse71':scores[71],'mean_rmse':np.mean(list(scores.values())),'gain42':gains[42],'gain71':gains[71]})
    comparison=pd.DataFrame(rows).sort_values('mean_rmse');comparison.to_csv(OUT/'selection.csv',index=False)
    eligible=comparison[(comparison.gain42>0)&(comparison.gain71>0)]
    winner=(eligible if len(eligible) else comparison).iloc[0]
    previous=json.loads((BASE/'improvement_v4/results/candidate_lock.json').read_text())
    contract={k:previous[k] for k in ['v2_configs','v2_weights','v3_parameters','v3_seeds']}
    w=float(winner.new_weight)
    contract.update(version='v5',v3_weight=1-w,seeds=[10242,20242],components={n:{'spec':SPECS[n],'weight':w*c} for n,c in combos[winner['name']].items()},
        development={'settings':len(SPECS),'shortlisted_settings':6,'splits':[42,71],'selection':winner.to_dict(),'screening_is_optimistic':True},
        confirmation={'new_splits':[18421,18422],'previous_splits':[2026,303],'city_stress':True,'all_splits_reuse_previously_seen_labels':True})
    (OUT/'candidate_lock.json').write_text(json.dumps(contract,indent=2)+'\n');print('LOCKED',json.dumps(contract['components'],indent=2),flush=True)

def confirm():
    train,test=load_data();y=train['Annual Turnover'].to_numpy(float);contract=load_contract();rows=[]
    for scheme in [18421,18422,2026,303,'city']:
        context={'contract':contract,'inputs':input_hashes(),'sources':source_hashes(),'scheme':str(scheme)}
        file=OUT/f'comparison_{scheme}.npz';meta=file.with_suffix('.json');start=time.time()
        if file.exists():
            saved=json.loads(meta.read_text());assert saved['context']==context
            assert saved['sha256']==hashlib.sha256(file.read_bytes()).hexdigest();p=dict(np.load(file))
        else:
            old=previous_module.checked_prior_v3(train,y,scheme,contract) if scheme in [2026,303,'city'] else None
            p=evaluate(train,y,scheme,contract,previous=old);np.savez_compressed(file,**p)
            meta.write_text(json.dumps({'context':context,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()},indent=2))
        row={'validation':str(scheme),'previous_rmse':rmse(y,p['previous']),'candidate_rmse':rmse(y,p['candidate'])}
        row['gain']=row['previous_rmse']-row['candidate_rmse'];row['seconds']=time.time()-start
        rows.append(row);pd.DataFrame(rows).to_csv(OUT/'confirmation.csv',index=False);print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','confirm']);a=p.parse_args()
    select() if a.mode=='select' else confirm()
