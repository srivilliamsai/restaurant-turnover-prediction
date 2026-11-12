"""Collect, independently verify and package the fixed exploratory rank check."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from core import ROOT, Engine, STEM, sha, save


def main():
    status=json.loads((ROOT/'transport_run/status.json').read_text())
    assert status['phase']=='complete' and status['owned_runtime_released']
    release=json.loads((ROOT/'account_release_check.json').read_text())
    assert release['status']=='PASS' and release['live_active_assignments']==0
    assert release['account_identity_sha256']==json.loads((ROOT/'authorization.json').read_text())['account_identity_sha256']
    for folder in ['results','delivery']:
        shutil.copytree(ROOT/'colab_results'/folder,ROOT/folder,dirs_exist_ok=True)
    e=Engine()
    assert e.sources==json.loads((ROOT/'source_frozen.json').read_text())['sources']
    original_archive=ROOT.parent/'improvement_v22/V22_Delivery.zip'
    assert sha(original_archive)==json.loads((ROOT.parent/'improvement_v22/archive_verification.json').read_text())['sha256']
    subprocess.run([sys.executable,str(ROOT/'verify_rank.py')],check=True,cwd=ROOT.parent)
    target=ROOT/'delivery'
    decision=json.loads((ROOT/'results/decision.json').read_text())
    review=json.loads((target/'independent_review.json').read_text())
    fresh=json.loads((target/'fresh_execution/fresh_execution_checks.json').read_text())
    snapshot=json.loads((ROOT/'leaderboard_snapshot.json').read_text())
    ledger=e.budget.state();ledger['supplement_complete']=True
    save(ROOT/'results/compute_budget.json',ledger)
    metrics=decision['metrics']
    positive=decision['descriptive_positive_robustness']
    verdict=('The fixed candidate is technically ready for an exploratory manual portal trial and has positive conditional robustness evidence. A move to 10th is possible but unconfirmed; ranks 9–6 are not established by these results.'
             if positive else
             'The fixed candidate is technically reproducible, but confirmation does not provide consistent positive robustness evidence. This candidate is not a supported route to a higher rank; a rank improvement remains unconfirmed.')
    summary={'status':'COMPLETE','verdict':verdict,'candidate':decision['choice'],
        'original_V22_development_gate_passed':False,'original_confirmation_gate_passed':decision['original_confirmation_gate_passed'],
        'descriptive_positive_robustness':positive,'mean_random_relative_gain':decision['mean_random_relative_gain'],
        'confirmation_metrics':metrics,'conditional_gain_CI95_INR':decision['uncertainty']['ci95_INR'],
        'fresh_notebook_reproduction':'PASS','maximum_CSV_difference_INR':fresh['maximum_final_difference_INR'],
        'independent_reconstruction':'PASS','confirmation_fold_outputs':review['complete_candidate_confirmation_folds'],
        'observed_portal_best_RMSE':11736110.7877709,'observed_rank_in_user_screenshot':11,
        'candidate_portal_score':None,'rank_change_established':False,'portal_uploads':0,
        'owned_runtime_released':True,'selected_account_active_assignments':0,
        'original_started_epoch':e.budget.authorization['started_epoch'],'elapsed_hours':ledger['seconds']/3600}
    save(target/'result_summary.json',summary)
    lines=['# V22 fixed-candidate rank check', '', '**'+verdict+'**', '',
        'Original V22 remains unqualified: its 0.123064% development gain was below the frozen 0.20% floor. The user subsequently asked to check all remaining steps. This separate, explicitly exploratory follow-up locks the already-selected P35-SC38 candidate at 100% replacement of the P35 share. No new model, seed, feature recipe or blend search was performed.', '',
        'The original four-hour clock, start time, paid-compute restriction and portal-upload restriction were retained. All original V22 files and its completed delivery archive remain unchanged.', '',
        '## Checked steps', '', '| Step | Result |', '|---|---|',
        '| Original inputs, scored artifacts and all inherited control assignments | Verified |',
        '| Exact V14 checkpoint/settings and 38-column feature projection | Verified |',
        '| Original engineering: repeated fit, row order, batching, individual rows, save/reload | Passed |',
        '| Current-runtime V9, V14 and fixed-candidate reproduction | Passed within INR 0.01 |',
        '| Two original development assignments, ten folds, three fixed blends | Verified; original 0.20% gate failed |',
        '| Three random confirmation assignments | '+('All improve' if all(metrics[str(s)]['gain']>0 for s in [48211,48212,48213]) else 'At least one does not improve')+' |',
        '| Stratified and city-group stress checks | '+('Within 1% deterioration limit' if all(metrics[s]['relative_gain']>=-.01 for s in ['city','strat:27421']) else 'Stress limit exceeded')+' |',
        '| Conditional paired-row uncertainty check | '+('Positive lower gain bound' if decision['uncertainty']['ci95_INR'][0]>0 else 'Lower gain bound is not positive')+' |',
        '| Original confirmation gate, including 0.20% mean gain | '+('Passed' if decision['original_confirmation_gate_passed'] else 'Failed')+'; original development failure remains |',
        '| Full-data fitting, exact 500 IDs/order/headers and finite nonnegative CSV | Passed |',
        '| Standalone notebook executed afresh and CSV reproduced | Passed |',
        '| Independent confirmation, bootstrap and export reconstruction | Passed |',
        '| Actual portal trial and rank | Not performed; unknown |', '',
        '## Measured confirmation', '', '| Assignment | V14 RMSE | Candidate RMSE | Gain in INR | Relative gain |',
        '|---|---:|---:|---:|---:|']
    for scheme,m in metrics.items():
        lines.append(f"| {scheme} | {m['v14_rmse']:,.2f} | {m['candidate_rmse']:,.2f} | {m['gain']:,.2f} | {m['relative_gain']:.6%} |")
    lo,hi=decision['uncertainty']['ci95_INR']
    lines += ['',f"Mean relative gain over the three random assignments: **{decision['mean_random_relative_gain']:.6%}**.",
              f'Conditional paired-row 95% gain interval: **INR {lo:,.2f} to {hi:,.2f}**, 5,000 draws at seed 8319.', '',
              'These historically examined labels are conditional robustness evidence. The interval does not account for the earlier selection history, every source of refitting variation or the hidden-test distribution. Local percentages cannot be multiplied by the portal score to predict a rank.', '',
              '## Leaderboard gaps from the supplied screenshot', '',
              '| Target | RMSE to beat | Required reduction from V14 | Relative reduction |',
              '|---|---:|---:|---:|']
    for rank,m in snapshot['targets'].items():
        lines.append(f"| {rank} | {m['score_to_beat']:,.2f} | {m['required_absolute_reduction']:,.2f} | {m['required_relative_reduction']:.6%} |")
    lines += ['', 'These are screenshot thresholds, not live leaderboard positions. The candidate has no portal score. The small gap to 10th makes an exploratory test relevant only if the user chooses to perform the remaining portal trial; it does not prove a gain. No numerical chance of success is estimated.', '',
        '## Reproduction and artifacts', '',
        f"Fresh notebook maximum CSV difference: INR {fresh['maximum_final_difference_INR']:.12g}.",
        'The V9 and V14 components were reproduced before the new blend. Original-Mac V7 state remains verified; identical fresh Linux V7 retraining is not claimed. Expected outputs were read only after fresh inference.', '',
        'EXPLORATORY-prefixed CSV and notebook files are review artifacts. They do not satisfy the original V22 development gate and have not been submitted. KEEP_V14 files are the exact observed-best fallback. Preserve the one-time final coding upload until the actual final model choice.', '',
        f"The owned runtime is released; a fresh account check found zero active assignments. Elapsed time including the original run, account change, this follow-up and delivery: {ledger['seconds']/3600:.4f} hours under the original four-hour ceiling.", '',
        'Run `python improvement_v22_rank_check/verify_rank.py` from the extracted archive root with the pinned NumPy/Pandas/scikit-learn/SciPy/CatBoost environment to reconstruct the evidence without model fits or GPU allocation.', '',
        'The final remaining measurement is one manual portal trial after checking actual portal availability and submission rules. Only a returned lower score can establish a higher position. No automatic upload was performed.']
    (target/'All_Steps_Review.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'README.md').write_text('# V22 exploratory rank check complete\n\n'+verdict+'\n\nSee delivery/All_Steps_Review.md, delivery/result_summary.json and delivery/independent_review.json. The original V22 gate remains failed. Candidate portal score and rank change are unknown. Free runtime released; zero active assignments. No portal upload.\n')
    for name in ['rank_protocol.json','rank_frozen.json','source_frozen.json','authorization.json','account_release_check.json','leaderboard_snapshot.json','README.md']:
        shutil.copy2(ROOT/name,target/name)
    members={}
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.py','.json','.md','.log'] and p.name not in ['archive_manifest.json','archive_verification.json']:
            members['improvement_v22_rank_check/'+p.name]=p
    for folder in ['results','delivery','reference','provenance','original_results','transport_run']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and '.partial' not in p.name and p.suffix!='.safetensors':
                members['improvement_v22_rank_check/'+str(p.relative_to(ROOT))]=p
    for name in e.inputs:members[name]=e.data/name
    save(ROOT/'archive_manifest.json',{'files_sha256':{n:sha(p) for n,p in members.items()},'portal_uploads':0,
         'original_V22_archive_sha256':sha(ROOT.parent/'improvement_v22/V22_Delivery.zip')})
    members['improvement_v22_rank_check/archive_manifest.json']=ROOT/'archive_manifest.json'
    archive=ROOT/'V22_Rank_Check_Delivery.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,p in members.items():z.write(p,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,p in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha(p)
    with tempfile.TemporaryDirectory(prefix='v22_rank_reproduction_',dir='/private/tmp') as temp:
        with zipfile.ZipFile(archive) as z:z.extractall(temp)
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT.parent/'.runtime_deps')
        p=subprocess.run([sys.executable,str(Path(temp)/'improvement_v22_rank_check/verify_rank.py')],
             cwd=temp,env=env,capture_output=True,text=True,timeout=120)
        assert p.returncode==0,p.stdout+p.stderr
        extracted=json.loads((Path(temp)/'improvement_v22_rank_check/delivery/independent_review.json').read_text())
        assert extracted==review
    save(ROOT/'archive_verification.json',{'status':'PASS','archive':archive.name,'sha256':sha(archive),
         'verified_files':len(members),'extracted_reconstruction':'PASS','original_V22_archive_unchanged':True,
         'portal_uploads':0})
    current=ROOT.parent/'CURRENT_BEST.md'
    text=current.read_text().replace('rank 9 in the user-supplied October 28 screenshot','rank 11 in the user-supplied October 29 19:44:46 screenshot')
    paragraph='\nV22 exploratory rank check: '+verdict+f" Mean random confirmation gain {decision['mean_random_relative_gain']:.6%}; original development qualification remains failed. Fresh notebook/CSV and independent reconstruction passed; no portal upload, scored V14 preserved, runtime released. See improvement_v22_rank_check/delivery/All_Steps_Review.md.\n"
    if paragraph not in text:current.write_text(text+paragraph)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
