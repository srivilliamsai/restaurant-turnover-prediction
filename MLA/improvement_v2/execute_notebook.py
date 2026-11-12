"""Run each saved Python cell and capture real stdout, stderr and figures."""
from pathlib import Path
import contextlib, io, json, base64, time, traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
NAME='submission'
path=ROOT/(NAME+'.ipynb')
nb=json.loads(path.read_text())
namespace={'__name__':'__main__'}
number=0
start=time.time()
for idx,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code': continue
    number+=1
    cell['execution_count']=number
    stdout,stderr=io.StringIO(),io.StringIO()
    figures=[]
    def capture_show(*args,**kwargs):
        for f in plt.get_fignums():
            fig=plt.figure(f)
            bio=io.BytesIO(); fig.savefig(bio,format='png',dpi=130,bbox_inches='tight')
            encoded=base64.b64encode(bio.getvalue()).decode()
            figures.append({'output_type':'display_data','metadata':{},'data':{'image/png':encoded,'text/plain':['<Figure rendered from executed notebook code>']}})
            plt.close(fig)
    plt.show=capture_show
    print(f'Executing cell {idx}, code #{number}',flush=True)
    try:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            exec(compile(''.join(cell['source']),f'notebook_cell_{idx}','exec'),namespace)
    except Exception as error:
        cell['outputs']=[{'output_type':'stream','name':'stdout','text':stdout.getvalue().splitlines(keepends=True)},
                         {'output_type':'error','ename':type(error).__name__,'evalue':str(error),'traceback':traceback.format_exc().splitlines()}]
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        raise
    cell['outputs']=[]
    if stdout.getvalue(): cell['outputs'].append({'output_type':'stream','name':'stdout','text':stdout.getvalue().splitlines(keepends=True)})
    if stderr.getvalue(): cell['outputs'].append({'output_type':'stream','name':'stderr','text':stderr.getvalue().splitlines(keepends=True)})
    cell['outputs'].extend(figures)
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print(f'Completed cell {idx}; elapsed {time.time()-start:.1f}s',flush=True)

table=namespace['validation_table']
avg=namespace['random_table'].drop(columns='Validation').mean()
weights=namespace['ensemble_weights']
report_lines=['# Restaurant turnover — model improvement report','',
              'Student: submission | Register number: submission | Team: submission','',
              '## Result','',
              f'The new candidate reduced mean RMSE across three fixed shuffled five-fold checks from {avg["Rank-3 model RMSE"]:,.2f} to {avg["New candidate RMSE"]:,.2f}, a reduction of {avg["Reduction"]:,.2f} ({100*avg["Reduction"]/avg["Rank-3 model RMSE"]:.3f}%).',
              '', 'The old CSV has a portal-reported RMSE of 12,204,615.6438253 and rank 3. The new CSV has not yet been scored on the portal. The reference leader score was 11,946,659.0702364. First place cannot be confirmed or promised from local validation.',
              '', '## Equal-condition validation comparison','',
              '| Validation | Rank-3 model | New candidate | RMSE reduction |',
              '|---|---:|---:|---:|']
for _,row in table.iterrows():
    report_lines.append(f'| {row["Validation"]} | {row["Rank-3 model RMSE"]:,.2f} | {row["New candidate RMSE"]:,.2f} | {row["Reduction"]:,.2f} |')
lo,hi=namespace['gain_ci']
report_lines += ['',f'The conditional 95% case-bootstrap interval for the RMSE reduction is [{lo:,.0f}, {hi:,.0f}]. This resamples validation errors; it does not include every source of model or distribution uncertainty.',
                 '', '## What changed','',
                 '- Compared 38 settings across CatBoost, LightGBM, XGBoost, Extra Trees, histogram boosting, Ridge, spline Ridge and SVR.',
                 '- Combined ten retained predictors with nonnegative weights summing to one; weights were fitted using development out-of-fold predictions with regularization.',
                 '- Added date, social, rating interaction and missingness features; tested dropping shifted service and comfort ratings.',
                 '- Tested Huber objectives, monotonic constraints and a cap applied only to training targets. Validation targets were never modified or removed.',
                 '- Used fixed tree counts and weights on subsequent validation. Seed 303 was reserved until the final candidate was fixed.',
                 '', '## Limits of the evidence','',
                 'The original notebook\'s selected validation score (19,854,618) used development-fold early stopping. The table above compares the old and new fixed final-model configurations on identical further splits; these are the relevant comparisons for this change. All splits reuse the same labeled data, so they are robustness estimates, not an untouched external test.',
                 '', 'The 33 training targets above INR 100 million account for about 57.5% of the mean-baseline squared error. The upper tail and the service/comfort distribution differences are possible reasons validation and portal scores differ. The hidden test labels are unavailable, so the cause cannot be confirmed.',
                 '', '## Execution and checks','',
                 f'- Executed all {number} Python cells; zero cell errors.',
                 '- Reproduced the old rank-3 CSV from its model within INR 0.01 tolerance.',
                 '- Checked the RMSE formula, input schema, target/ID exclusion, train/validation disjointness, city-group isolation, and one validation prediction per row.',
                 '- Checked shuffled prediction-row alignment, unseen/missing feature values, finite nonnegative predictions, 500 unique restaurant IDs in test order, exact headers, 501 CSV lines and CSV round-trip.',
                 '', '## Submission','',
                 f'Upload `{NAME}.csv` to the hackathon module. The new file is a candidate whose hidden-test performance must be measured by the portal. Keep the best observed submission.',
                 f'Upload the executed `{NAME}.ipynb` under LLJ_Oct_2026 only when your model choice is final. The assignment is described as allowing one submission.',
                 'The rank-3 files are preserved in `improvement_v2/rank3_backup`. The student register number belongs in the filenames; the CSV row IDs remain the restaurant IDs from Test_dataset.csv.',
                 '', 'The portal screenshots show team name submission. Faculty instructions require a Registernumber_Name registration format; filenames do not update portal registration. Confirm that requirement with the portal settings or Program Manager before the deadline.']
(ROOT/'improvement_report.md').write_text('\n'.join(report_lines)+'\n')
summary=dict(mean_old_rmse=float(avg['Rank-3 model RMSE']),mean_new_rmse=float(avg['New candidate RMSE']),mean_improvement=float(avg['Reduction']),weights=weights,validation=table.to_dict(orient='records'),bootstrap_interval=[float(lo),float(hi)],notebook_code_cells=number,elapsed_seconds=time.time()-start)
(ROOT/'results/final_execution.json').write_text(json.dumps(summary,indent=2))
print('ALL CELLS EXECUTED',json.dumps(summary),flush=True)
