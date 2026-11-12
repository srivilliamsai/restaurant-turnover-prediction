from pathlib import Path
import json, shutil, hashlib
import pandas as pd
import numpy as np
import nbformat

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
STEM='submission'
summary=json.loads((ROOT/'results/notebook_execution.json').read_text())
book=nbformat.read(ROOT/(STEM+'.ipynb'),as_version=4);nbformat.validate(book)
assert len([c for c in book.cells if c.cell_type=='code'])==summary['code_cells']
for c in book.cells:
    if c.cell_type=='code':
        assert c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs)
        compile(c.source,'notebook','exec')
csv_path=ROOT/(STEM+'.csv');df=pd.read_csv(csv_path)
test=pd.read_csv('~/Downloads/Test_dataset.csv')
assert df.shape==(500,2) and df.columns.tolist()==['Registration Number','Annual Turnover']
assert df['Registration Number'].equals(test['Registration Number'])
assert np.isfinite(df['Annual Turnover']).all() and (df['Annual Turnover']>=0).all()
assert len(csv_path.read_text().splitlines())==501

means=summary['means']; table=pd.DataFrame(summary['validation'])
gain=means['Submitted model RMSE']-means['Current candidate RMSE']
old_score=12204615.6438253;leader=11946659.0702364
lo,hi=summary['bootstrap_interval']
lines=[
'# Restaurant turnover — October 12 follow-up',
'', '**Student register number:** submission', '', '**Student name:** submission', '', '**Team name:** submission',
'', '## What the screenshot establishes',
'', 'The October 12 screenshot shows one submission, RMSE **12,204,615.6438253**, and **rank 7**. This is the same score as the October 11 rank-3 submission. The rank changed because competing entries moved ahead. The screenshot does not show a submission result for either improved candidate.',
'', f'The leader shown has RMSE **{leader:,.2f}**. The original submission needs a reduction of more than **{old_score-leader:,.2f}** ({100*(old_score-leader)/old_score:.2f}% of its RMSE) to beat that score, assuming it remains unchanged. No new portal score or rank is claimed.',
'', '## New local comparison',
'', 'All rows below use original INR RMSE and retain every validation target. The two improved models and the originally submitted model use identical folds for each comparison.',
'', '| Validation | Originally submitted | Previous improved (v2) | Current candidate | Gain vs v2 |',
'|---|---:|---:|---:|---:|']
for r in summary['validation']:
    label='City groups (stress test)' if r['Validation']=='city' else 'Shuffled 5-fold seed '+r['Validation']
    lines.append(f"| {label} | {r['Submitted model RMSE']:,.2f} | {r['Previous improved RMSE']:,.2f} | {r['Current candidate RMSE']:,.2f} | {r['Gain vs previous improved']:,.2f} |")
lines += [f"| Mean of four shuffled assignments | {means['Submitted model RMSE']:,.2f} | {means['Previous improved RMSE']:,.2f} | {means['Current candidate RMSE']:,.2f} | {means['Gain vs previous improved']:,.2f} |",
'', f'The current candidate lowers average shuffled-fold RMSE by **{gain:,.2f} ({100*gain/means["Submitted model RMSE"]:.3f}%)** versus the originally submitted model, and by **{means["Gain vs previous improved"]:,.2f}** versus v2.',
'', '**Limits and tradeoff:** the city-isolation check is **36,723.89 worse than v2**. The conditional 95% row-bootstrap interval for the gain versus v2 is '+f'**[{lo:,.2f}, {hi:,.2f}]**, which includes zero. The incremental gain is not decisive under that check. The candidate is supported by four positive random-split comparisons, but it is not a proven improvement on unseen cities or on the hidden test set.',
'', 'These assignments reuse the same labeled data used during development. Seed 909 was reserved for this round’s split comparison, but its restaurants were not an untouched holdout throughout all previous work. Scores are robustness evidence, not an unbiased external-test or nested model-selection estimate. Bootstrap uncertainty is conditional on the fitted models and does not cover all selection or distribution uncertainty.',
'', '## What changed',
'', '- Evaluated 42 additional model settings: 14 CatBoost settings averaged over three seeds, 20 spline/linear/kernel settings, and 8 Random Forest settings, plus coarse blend comparisons.',
'- Fixed the candidate after seed-42 development selection: 50% of the previous ten-model ensemble plus 50% of a three-seed CatBoost model using Bernoulli row subsampling.',
'- New component: depth 5, 850 trees, learning rate 0.04, Huber delta INR 40 million, L2 10, subsample 0.7; seeds 1042, 2042 and 3042 for full fitting.',
'- Random Forest settings all worsened development blend RMSE. The best smooth-model addition improved it by only about 3,085, so it was excluded.',
'- All validation targets retain their actual original values. Training-only robust losses and capping do not redefine the metric.',
'', '## Execution and file checks',
'', f'- Executed all {summary["code_cells"]} code cells of the new notebook with zero saved errors.',
'- The experimental scripts trained the new fold models. The local notebook transparently reused hash-verified actual fold predictions and recalculated the metrics; its input hashes, model-configuration hash and file hashes were checked.',
'- All final models were refitted inside the notebook. Those predictions reproduced the independently generated candidate within INR 0.01. In Colab, no local cache exists and the included code trains the validation models.',
'- Verified target/ID exclusion, original-scale RMSE, fold coverage and separation, canonical-city group separation, prediction row permutation, unseen and missing features, and exact CSV round-trip.',
'- Checked 500 data rows plus one header, two exact column names, unique restaurant IDs in original test order, finite nonnegative predictions and no added index column. Deliberately invalid submissions were rejected.',
'- Reviewed both saved diagnostic plots. The notebook includes code, actual outputs and inferences.',
'', '## Files and next action',
'', f'- Current CSV: `{STEM}.csv` in the main MLA folder.',
f'- Matching executed notebook: `{STEM}.ipynb` in the main MLA folder.',
'- Previous improved files are preserved in `improvement_v2/` and `improvement_v3/previous_delivery_backup/`. The originally submitted files remain in `improvement_v2/rank3_backup/`.',
'', 'Upload the current CSV under **assignment M.Tech. MLA Hackathon -Restaurant Turnover Prediction** and record the actual returned RMSE. A hidden-test score cannot be calculated locally because the test targets are not provided. Upload the final executed notebook under **LLJ_Oct_2026** when your model choice is final; the stated assignment rule permits only one notebook submission.',
'', 'CSV row registration numbers identify restaurants. The student number is used in the filenames and notebook cover. The faculty’s portal-registration naming requirement is separate from the submission-file format.',
'', '**Current CSV SHA256:** `'+hashlib.sha256(csv_path.read_bytes()).hexdigest()+'`']
(ROOT/'improvement_report.md').write_text('\n'.join(lines)+'\n')

# Preserve the exact current deliverables before replacing them; fail on an
# unexpected user edit instead of silently overwriting it.
before=json.loads((ROOT/'results/submission_checks.json').read_text())
assert hashlib.sha256((BASE/(STEM+'.csv')).read_bytes()).hexdigest()==before['csv_sha256']
assert hashlib.sha256((BASE/(STEM+'.ipynb')).read_bytes()).hexdigest()==before['notebook_sha256']
backup=ROOT/'previous_delivery_backup';backup.mkdir(exist_ok=True)
for extension in ['.csv','.ipynb']:
    shutil.copy2(BASE/(STEM+extension),backup/(STEM+extension))
    shutil.copy2(ROOT/(STEM+extension),BASE/(STEM+extension))
manifest={'version':'October 12 candidate v3','portal_score':None,'last_known_portal_score':old_score,'last_known_rank':7,
          'files':{extension:{'path':str(BASE/(STEM+extension)),'sha256':hashlib.sha256((BASE/(STEM+extension)).read_bytes()).hexdigest()} for extension in ['.csv','.ipynb']},
          'code_cells':summary['code_cells'],'data_rows':500,'columns':df.columns.tolist(),'mean_validation':means,
          'caveats':['City stress test worsens versus v2','Incremental bootstrap gain interval includes zero','No new portal result available']}
(ROOT/'results/delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
