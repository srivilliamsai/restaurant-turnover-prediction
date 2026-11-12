"""Record the user's portal screenshot without changing a scored predictor."""
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

root=Path(__file__).resolve().parent
out=root/'colab_cli_runs/20261026T160148Z/portable_delivery'
stem='submission'
csv_path=out/'submission'/(stem+'.csv')
notebook=out/'submission'/(stem+'.ipynb')
screenshot=Path('~/Documents/screenshots/Screenshot 2026-10-27 at 12.25.20\u202fAM.png')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(csv_path)=='5a384dc3d8bf58c4e29f0d6cac646aa33afe22bdcdd186cccd9758342769bbd4'
manifest_path=out/'manifest.json'
manifest=json.loads(manifest_path.read_text())
notebook_hash=sha(notebook)
assert notebook_hash==manifest['files'][str(notebook.relative_to(out))]
old,new=12024316.5355221,11904515.9864996
record={'version':'v9_trial_1','portal_rmse':new,'rank_in_screenshot':9,
        'previous_best_portal_rmse':old,'rmse_reduction':old-new,
        'relative_reduction_percent':100*(old-new)/old,
        'csv':str(csv_path),'csv_sha256':sha(csv_path),
        'matching_notebook':str(notebook),'matching_notebook_sha256':notebook_hash,
        'source':'User-provided portal result screenshot after stating they would upload V9',
        'screenshot':str(screenshot),'screenshot_sha256':sha(screenshot),
        'screenshot_timestamp_local':'2026-10-27T00:25:20+05:30',
        'timezone':'Asia/Calcutta','submission_time':None,
        'attribution':'Associated with the recommended V9 file from the preceding upload instructions and user response. The screenshot does not independently expose the uploaded file hash.',
        'agent_performed_upload':False,'recommended_current_submission':'v9_trial_1',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat()}
result=root/'portal_result_2026-10-27.json'
result.write_text(json.dumps(record,indent=2)+'\n')
shutil.copyfile(result,out/result.name)
shutil.copyfile(screenshot,out/'portal_result_2026-10-27.png')
backup=root/'scored_v9_backup';backup.mkdir(exist_ok=True)
for path in [csv_path,notebook]:
    target=backup/path.name
    if target.exists():assert sha(target)==sha(path)
    else:shutil.copyfile(path,target)
shutil.copyfile(result,backup/result.name)
trials=root/'portal_trials.csv'
with trials.open(newline='') as f:
    reader=csv.DictReader(f);fieldnames=reader.fieldnames;rows=list(reader)
for row in rows:
    if row['version']=='v7_trial_1':row['status']='previous best; rank 24 in original October 26 screenshot'
rows=[row for row in rows if row['version']!='v9_trial_1']
rows.append({'version':'v9_trial_1','csv_sha256':sha(csv_path),'portal_rmse':str(new),'rank':'9','status':'user-uploaded; screenshot-confirmed score; current best'})
with trials.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
shutil.copyfile(trials,out/'portal_trials.csv')
report=out/'V9_Report.md'
text=report.read_text().replace('**One validation-qualified V9 CSV is ready for a portal trial. Current observed portal best remains V7: 12,024,316.5355221.**',
    '**V9 is the current observed portal best: RMSE 11,904,515.9864996; rank 9 in the user-provided October 27 screenshot.**')
text=text.replace('No paid compute or portal upload was performed.','No paid compute was used. The user uploaded V9; the agent performed no portal upload.')
section='''
## Observed portal result — October 27

The user supplied a result screenshot showing RMSE **11,904,515.9864996**, previous best **12,024,316.5355221**, and rank **9**. This improves RMSE by **119,800.55 (0.9963%)**. V9 is now the recommended current submission.

CSV SHA256: `5a384dc3d8bf58c4e29f0d6cac646aa33afe22bdcdd186cccd9758342769bbd4`.

The score-to-file association follows the preceding V9 upload instructions and user response; the screenshot itself does not show the filename or hash. The CSV and executed notebook remain byte-identical to the verified delivery and are preserved in `scored_v9_backup`. No new model fitting or parameter selection was performed after observing this score. The executed notebook retains its pre-submission narrative as a dated execution snapshot.
'''
if '## Observed portal result — October 27' not in text:text+='\n'+section
report.write_text(text)
readme=out/'READ_ME_FIRST.md'
text=readme.read_text()
banner='''**Confirmed current best: V9, RMSE 11,904,515.9864996; rank 9 in the October 27 screenshot.** Keep its matching CSV and executed notebook. The instructions below describe the original trial/reproduction procedure; the portal result has now been recorded.\n\n'''
if 'Confirmed current best: V9' not in text:text=text.replace('# Using the verified V9 trial\n\n','# Using the verified V9 trial\n\n'+banner)
readme.write_text(text)
manifest['observed_portal_best']={'version':'V9','trial':'v9_trial_1','rmse':new,'rank_in_screenshot':9,'csv_sha256':sha(csv_path)}
manifest['portal_uploads_by_agent']=0
manifest['observed_user_portal_trials']=1
manifest['portal_uploads_field_scope']='Legacy portal_uploads counts uploads performed by the agent.'
for p in [report,readme,out/result.name,out/'portal_result_2026-10-27.png',out/'portal_trials.csv']:
    manifest['files'][str(p.relative_to(out))]=sha(p)
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
archive=root/'V9_Portable_Delivery.zip'
for relative,expected in manifest['files'].items():assert sha(out/relative)==expected,relative
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for relative in sorted(manifest['files']):z.write(out/relative,relative)
    z.write(manifest_path,'manifest.json')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
status_path=out/'delivery_status.json'
status=json.loads(status_path.read_text());status['archive_sha256']=sha(archive)
status['observed_portal_best']=manifest['observed_portal_best']
status['observed_user_portal_trials']=1
status_path.write_text(json.dumps(status,indent=2)+'\n')
best_text=f'''# Current best observed submission

**V9 Trial 1 — portal RMSE 11,904,515.9864996; rank 9 in the October 27 screenshot.**

Previous best V7: 12,024,316.5355221. Reduction: 119,800.55 (0.9963%).

- [Scored CSV]({backup/(stem+'.csv')})
- [Matching executed notebook]({backup/(stem+'.ipynb')})
- [Score record and attribution]({result})
- [Measured report]({report})
- [Updated delivery package]({archive})

CSV SHA256: `5a384dc3d8bf58c4e29f0d6cac646aa33afe22bdcdd186cccd9758342769bbd4`.

The user performed the upload. Score and rank are confirmed by the supplied screenshot; file attribution follows the preceding V9 upload context. The CSV and notebook are unchanged. Keep the one-time coding upload pending the final model choice.
'''
(root.parent/'CURRENT_BEST.md').write_text(best_text)
(root/'CLI_PROGRESS.md').write_text(best_text)
assert sha(notebook)==notebook_hash
print(json.dumps({'portal_rmse':new,'rank':9,'improvement':old-new,'improvement_percent':100*(old-new)/old,'scored_csv_and_notebook_unchanged':True,'result':str(result),'package_sha256':sha(archive)},indent=2))
