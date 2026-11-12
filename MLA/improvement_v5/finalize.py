"""Publish measured results and promote only under the predeclared rule."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
import numpy as np
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';BASE=ROOT.parent
STEM='submission'
portal=json.loads((ROOT/'portal_snapshot.json').read_text())
contract=json.loads((OUT/'candidate_lock.json').read_text())
execution=json.loads((OUT/'notebook_execution.json').read_text())
uncached=json.loads((OUT/'uncached_notebook_check.json').read_text())
assert execution['full_fit_reproduced'] and uncached['passed']
scores=pd.read_csv(OUT/'confirmation.csv',dtype={'validation':str})
random=scores[scores.validation!='city'];fresh=scores[scores.validation.isin(['18421','18422'])]
gain=float(random.gain.mean());fresh_gain=float(fresh.gain.mean())
supported=bool(gain>0 and fresh_gain>0)
interval=execution['bootstrap_interval']
status='Candidate recommended for a new portal trial' if supported else 'Experimental candidate; retain confirmed v3 as preferred'
summary=f'''## Measured result of this executed local run

**{status}.** Mean RMSE reduction across four shuffled assignments: **{gain:,.2f}**.
Across the two new split assignments: **{fresh_gain:,.2f}**.
Conditional bootstrap interval: **[{interval[0]:,.2f}, {interval[1]:,.2f}]**.

One complete five-fold assignment was also retrained through this notebook's
embedded code without prediction caches. Its predictions matched the experiment
reference within INR0.01. All selected full-data models were freshly fitted in
this notebook, and the resulting CSV matched the independent full fit.

These are local validation results. The candidate has no observed portal score,
and first place is not established. The confirmed prior portal score remains
12,094,600.3216516; the October18 leading score is 11,917,847.2693885.
'''
path=ROOT/(STEM+'.ipynb');book=nbformat.read(path,as_version=4)
book.cells=[c for c in book.cells if not (c.cell_type=='markdown' and c.source.startswith('## Measured result of this executed local run'))]
book.cells.append(nbformat.v4.new_markdown_cell(summary));nbformat.validate(book);nbformat.write(book,path)
subprocess.run([sys.executable,str(ROOT/'check_submission.py')],check=True)
checks=json.loads((OUT/'submission_checks.json').read_text());assert checks['passed']
for kind,item in portal['files'].items():
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
if supported:
    for extension in ['csv','ipynb']:shutil.copy2(ROOT/(STEM+'.'+extension),BASE/(STEM+'.'+extension))
else:
    for kind,item in portal['files'].items():
        assert hashlib.sha256((BASE/(STEM+'.'+kind)).read_bytes()).hexdigest()==item['sha256']

def table(frame):
    lines=['| Validation | Previous v3 RMSE | New v5 RMSE | Reduction (positive is better) |','|---|---:|---:|---:|']
    for row in frame.itertuples():lines.append(f'| {row.validation} | {row.previous_rmse:,.2f} | {row.candidate_rmse:,.2f} | {row.gain:+,.2f} |')
    return '\n'.join(lines)

components='\n'.join(f'- **{c["weight"]:.1%} {name}**: `{json.dumps(c["spec"],sort_keys=True)}`.' for name,c in contract['components'].items())
artifact_checks=checks['checks']['csv'];full=json.loads((OUT/'full_fit.json').read_text())
local_mean=float(random.previous_rmse.mean());candidate_mean=float(random.candidate_rmse.mean())
candidate_link=str(ROOT/(STEM+'.csv'));notebook_link=str(ROOT/(STEM+'.ipynb'))
report=f'''# Restaurant turnover — October 18 improvement report

**Student register number:** submission  
**Student name:** submission  
**Team name:** submission

## Decision

**{status}.** {'The workspace root CSV and notebook now contain this tested v5 candidate. The portal-confirmed v3 remains in confirmed_best_backup.' if supported else 'The workspace root files remain the portal-confirmed v3. The v5 files below are experimental.'}

The new CSV is a model-generated candidate, not a confirmed first-place submission.
No upload was performed. Only the portal can measure its hidden-test RMSE.

## What changed on the leaderboard

Source: your October18, 12:42PM screenshots; this is not a live leaderboard query.

| Entry | Rank in screenshot | Portal RMSE |
|---|---:|---:|
| submission — your previous submitted model | 18 | 12,094,600.3216516 |
| submission — leader | 1 | 11,917,847.2693885 |
| New v5 candidate | Unknown | Not submitted |

Your score did not worsen; competitors moved ahead. There are still two recorded
submissions. Beating the shown leader requires a reduction greater than
**176,753.05 (1.4614%)**, provided that leading score remains unchanged.

## Model and experiment design

Screened **22 settings**, each with five-fold validation and two model seeds.
Six settings advanced to a second development assignment. Compared 24 coarse
blend combinations, then froze the selected settings before confirmation.

- **{contract['v3_weight']:.1%} previous v3 ensemble**.
{components}
- Each new component averages two CatBoost model seeds, 10242 and 20242.
- No hidden test labels, external target matching, or adjustments fitted to portal scores.
- Training-only target transformations, caps and feature selection were compared.
  Every validation score uses all original turnover labels in INR. Encoders and
  imputers are fitted only on the corresponding training fold.

Full configuration: [candidate_lock.json]({OUT/'candidate_lock.json'}).
All development comparisons: [selection.csv]({OUT/'selection.csv'}).

## Same-fold validation results

{table(scores)}

| Shuffled-validation summary | Previous | Candidate | Reduction |
|---|---:|---:|---:|
| Mean of four assignments | {local_mean:,.2f} | {candidate_mean:,.2f} | {gain:+,.2f} |
| Mean of two new assignments | {fresh.previous_rmse.mean():,.2f} | {fresh.candidate_rmse.mean():,.2f} | {fresh_gain:+,.2f} |

18421 and 18422 are new split assignments. 2026 and 303 were used in earlier
rounds; city groups form a separate stress check and are excluded from the random
split average. All assignments reuse previously examined labels, so they are
robustness checks rather than an untouched final test. Known city aliases are
grouped, but unrecognized spelling variants may remain.

Conditional row-bootstrap 95% interval for the RMSE reduction:
**[{interval[0]:,.2f}, {interval[1]:,.2f}]**.
{'It includes zero: the improvement is uncertain.' if interval[0]<=0<=interval[1] else 'It excludes zero conditional on these fitted fold predictions.'}
The interval does not include the uncertainty from all model selection or a
different test distribution. Local validation RMSE must not be subtracted from
the portal score to predict ranking.

![Validation comparison]({OUT/'notebook_figure_2.png'})

## Execution and tests

- {execution['code_cells']} notebook code cells executed, with saved outputs and {execution['figures']} figures; no saved errors.
- Actual hash-verified fold predictions reused explicitly in the local executed notebook; metrics recomputed.
- The notebook's default uncached implementation retrained all five folds of assignment18421 and reproduced the reference predictions within INR0.01.
- Full selected models freshly fitted in both the independent pipeline and notebook; matching predictions.
- The predecessor refit reproduces the portal-confirmed v3 CSV within INR0.01.
- Predictions remain aligned when test rows are permuted. Missing values and an unseen city were checked.
- Independent and pipeline validators rejected 12 malformed CSV cases: missing/extra row, row order, headers/order, index column, student ID replacement, duplicate IDs, NaN, both infinities and negative turnover.
- Exactly **500 rows, two columns, 501 CSV records including header**, original test IDs/order, finite positive turnover, no extra index.
- Submission range: **{artifact_checks['min']:,.2f}–{artifact_checks['max']:,.2f}**; mean absolute change from v3: **{full['mean_absolute_prediction_change']:,.2f}**.

Details: [submission_checks.json]({OUT/'submission_checks.json'}),
[uncached_notebook_check.json]({OUT/'uncached_notebook_check.json'}).

## Files and use

1. [New candidate CSV]({candidate_link}) — `{STEM}.csv`.
2. [Matching executed Colab notebook]({notebook_link}) — `{STEM}.ipynb`.
3. [Portal-confirmed v3 backup]({portal['files']['csv']['path']}).

To reproduce in Colab, upload the notebook and run all cells on CPU. Upload
Train_dataset.csv, Test_dataset.csv, Sample_Solution.csv and Data_Dictionary.csv
when prompted. Colab trains validation folds by default; the final cell downloads
the CSV. The original source data is not embedded in the notebook.

If trying the candidate, upload the CSV under the hackathon. Save the returned
RMSE alongside this version. The row-level Registration Number column contains
the restaurant IDs, not your student register number. Submit the executed notebook
matching your final choice to LLJ_Oct_2026 when ready; the supplied instructions
allow one final notebook submission.

CSV SHA256: `{checks['checks']['csv']['sha256']}`  
Notebook SHA256: `{checks['checks']['notebook']['sha256']}`
'''
(ROOT/'REPORT.md').write_text(report)
delivery={'status':status,'recommended_for_trial':supported,'root_promoted':supported,
          'current_portal_rmse':portal['current_rmse'],'leader_portal_rmse':portal['leader_rmse'],
          'candidate_portal_rmse':None,'mean_local_reduction':gain,'new_split_mean_reduction':fresh_gain,
          'bootstrap_interval':interval,'csv':candidate_link,'notebook':notebook_link,'report':str(ROOT/'REPORT.md'),
          'csv_sha256':checks['checks']['csv']['sha256'],'notebook_sha256':checks['checks']['notebook']['sha256']}
(OUT/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n');print(json.dumps(delivery,indent=2))
