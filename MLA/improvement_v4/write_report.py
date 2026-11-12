"""Summarize measured results without turning local RMSE into a portal forecast."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
STEM='submission'
portal=json.loads((ROOT/'confirmed_portal_result.json').read_text())
contract=json.loads((OUT/'candidate_lock.json').read_text())
table=pd.read_csv(OUT/'confirmation.csv',dtype={'validation':str})
assert len(table)==7,'Finish all validation runs before writing report'
y=pd.read_csv('~/Downloads/Train_dataset.csv')['Annual Turnover'].to_numpy(float)
random=table[table.validation!='city'];fresh=table[table.validation.isin(['1701','1711'])]
means=random[['v3_rmse','v4_rmse','gain_vs_v3']].mean()
freshmeans=fresh[['v3_rmse','v4_rmse','gain_vs_v3']].mean()
errors=[]
for seed in [1701,1711,71,2026,303,909]:
    p=np.load(OUT/f'comparison_{seed}.npz')
    errors.append(np.column_stack([(y-p['previous'])**2,(y-p['candidate'])**2]))
sq=np.mean(errors,axis=0);rng=np.random.default_rng(812)
gain=[]
for _ in range(2000):
    idx=rng.integers(0,len(y),len(y));sample=np.sqrt(sq[idx].mean(axis=0));gain.append(sample[0]-sample[1])
ci=np.quantile(gain,[.025,.975])
city=table[table.validation=='city'].iloc[0]
stronger=bool(means['gain_vs_v3']>0 and freshmeans['gain_vs_v3']>0)
result={'six_random_means':means.to_dict(),'two_new_split_means':freshmeans.to_dict(),
        'city_gain':float(city.gain_vs_v3),'conditional_bootstrap_95pct_gain_interval':ci.tolist(),
        'random_wins':int((random.gain_vs_v3>0).sum()),'random_splits':6,
        'supported_for_trial':stronger,'candidate_portal_rmse':None}
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=[]
for _,r in table.iterrows():
    label={'1701':'1701 — new split','1711':'1711 — new split','city':'City groups — stress'}.get(r.validation,r.validation)
    lines.append(f'| {label} | {r.v3_rmse:,.2f} | {r.v4_rmse:,.2f} | {r.gain_vs_v3:+,.2f} |')
recommend=('The candidate improves both the six-split average and the average of the two newly reserved split assignments. It is reasonable to try as a new hackathon CSV submission, with the uncertainty below.' if stronger else 'The candidate does not improve both the overall and newly reserved split averages. Keep the portal-confirmed v3 file as the preferred model. The v4 file is an experimental alternative, not a verified improvement.')
weights='\n'.join(f'- {100*c["weight"]:g}% {name}, averaging seeds {contract["seeds"]}.' for name,c in contract['components'].items())
report=f'''# Restaurant turnover: fourth improvement round

**Student register number:** submission  
**Student name:** submission  
**Team name:** submission

## Actual hackathon results

These values come from the supplied screenshots, not from local validation.

| Submission | Portal RMSE | Rank shown |
|---|---:|---:|
| Original v1 | {portal['previous_portal_rmse']:,.7f} | 7 before the latest upload |
| Successful v3, Oct 12 at 10:22 PM | {portal['latest_portal_rmse']:,.7f} | 3 |
| Leader shown, Aathan | {portal['leader_rmse']:,.7f} | 1 |
| New v4 candidate | **Unknown until uploaded** | Unknown |

The successful upload reduced RMSE by **{portal['improvement']:,.2f}**. It still needs a reduction greater than **{portal['leader_gap']:,.2f}**, or about **{100*portal['leader_gap']/portal['latest_portal_rmse']:.3f}%**, to beat the leader shown if that leader score is unchanged. No rank guarantee is possible from local validation.

## What was changed

The round screened **24 settings**: 12 numeric-model parameter variants, 10 feature representations including the base comparison, and two feature-based training-weight variants. It compared **78 coarse blends**, including two equal-weight family combinations, using five-fold seed 42 only. The selected formula was frozen before the checks below:

- {100*contract['v3_weight']:g}% of the exact successful v3 ensemble.
{weights}

The numeric variant uses 32 candidate split borders. The categorical variant treats 18 integer rating fields as categories. Both use depth 5, 850 trees, learning rate 0.04, Huber delta INR 40 million, L2 10 and Bernoulli subsampling 0.7. The 18 rating fields had no unseen test categories or string-format collisions. New components use model seeds 1042/2042, plus the fold index during validation.

Development RMSE changed from **{contract['development']['baseline_rmse']:,.2f}** to **{contract['development']['candidate_rmse']:,.2f}**. This score was used for selection and is optimistic evidence. We did not keep searching or retune the weights after seeing confirmation scores.

## Same-fold local RMSE comparison

All 3,493 original training targets are scored in original INR units. Lower is better. A positive reduction means improvement.

| Validation assignment | Successful v3 model | New v4 candidate | RMSE reduction |
|---|---:|---:|---:|
{chr(10).join(lines)}
| Mean of six shuffled assignments | {means['v3_rmse']:,.2f} | {means['v4_rmse']:,.2f} | {means['gain_vs_v3']:+,.2f} |
| Mean of the two new assignments | {freshmeans['v3_rmse']:,.2f} | {freshmeans['v4_rmse']:,.2f} | {freshmeans['gain_vs_v3']:+,.2f} |

The candidate improves **{int((random.gain_vs_v3>0).sum())} of six** shuffled assignments. City groups are a separate stress test and are excluded from their average. The city comparison uses the existing canonical city aliases; other spelling variants may still represent the same city, so it is not a perfect geographic holdout.

## Interpretation and limitations

{recommend}

The conditional row-bootstrap 95% interval for the RMSE reduction is **[{ci[0]:,.2f}, {ci[1]:,.2f}]**. {'It includes zero, so an incremental improvement is not established by this check.' if ci[0]<=0<=ci[1] else 'It does not include zero for these fitted predictions, but does not cover model-selection and distribution uncertainty.'} The interval resamples existing prediction errors; it does not refit or repeat the whole search.

All split assignments reuse the same labeled restaurants, including labels used in earlier development. Seeds 1701/1711 were reserved during this round, but this is not an untouched external test or nested evaluation of all earlier searches. The portal evaluates different restaurants. Local and portal RMSE therefore need not match, and their difference cannot be subtracted from a new local score to predict rank.

Validation rows and targets were never deleted, clipped or log-transformed. Huber loss and the inherited training-only cap affect model training only. No hidden targets or reconstructed labels were used. Test feature missingness informed two explicitly evaluated weighting hypotheses, which were not selected.

## Files, execution and checks

- `{STEM}.csv`: exactly 500 records, two exact headers, original test restaurant IDs and row order, finite nonnegative predictions, no index column.
- `{STEM}.ipynb`: self-contained Colab notebook with the same frozen model and checks. Student identity appears on its cover and in filenames; row IDs remain restaurant IDs.
- The original successful v3 files are preserved in `rank3_confirmed_backup/` with their hashes and screenshot score in `confirmed_portal_result.json`.
- The independent full fit reproduces v3's submitted predictions to numerical precision before constructing v4.
- The executed notebook transparently reuses hash-verified fold predictions, recomputes RMSE/figures, and reruns all full-data fits and behavior/export checks. Colab defaults to retraining validation folds. A separate uncached five-fold notebook run checks that default path.
- Invalid headers, row count/order, extra index, duplicates, student IDs, NaN, infinity and negative predictions are tested for rejection. Predictions are also checked under row permutation and missing/unseen feature values.

**Completed execution checks:** all 10 notebook code cells ran with no saved errors and produced two figures. The independent file checker rejected 12 deliberately invalid submissions. The separate uncached five-fold run on split 1701 passed with **0.0 INR maximum prediction difference** for both the confirmed model and the new candidate. Execution logs and machine-readable results are in `results/`. Candidate portal accuracy is unmeasured.

### Download the intended version

- [Confirmed best CSV — v3]({ROOT.parent / (STEM+'.csv')}) and [matching executed notebook]({ROOT.parent / (STEM+'.ipynb')}): these are the preserved model files associated with the observed rank-3 score. The notebook was executed before that portal result; its original pending-result statements are historical.
- [Experimental v4 CSV]({ROOT / (STEM+'.csv')}) and [matching executed Colab notebook]({ROOT / (STEM+'.ipynb')}): these contain this round's unscored candidate and its complete comparison results. Both versions use the required filenames in separate folders.

## Submission

If you choose to test this experimental v4 CSV, upload it to **assignment M.Tech. MLA Hackathon -Restaurant Turnover Prediction** and keep its returned score with this exact version. Until a better portal result is observed, retain the confirmed v3 files. Submit the executed notebook matching your final chosen model to **LLJ_Oct_2026**; the supplied instructions say this assignment allows one submission. A filename change does not alter the portal registration name.
'''
(ROOT/'improvement_report.md').write_text(report)
print(json.dumps(result,indent=2))
