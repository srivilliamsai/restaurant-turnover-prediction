"""Record the human-approved 24-hour cap without changing model/cache sources."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1];out=root/'colab_results/results'
receipt=out/'budget_amendment.json';ledger=out/'compute_budget.json'
if not receipt.exists():
    previous=ledger.read_bytes();record=json.loads(previous)
    prior=float(record['seconds']);remaining=24-prior/3600
    assert 0<remaining<=18
    history=out/'budget_history';history.mkdir(exist_ok=True)
    (history/'original_18h_ledger.json').write_bytes(previous)
    evidence={'authorized_by':'Direct human reply to the budget question in this chat',
              'authorization':'Raise total cap to 24 hours and finish V10',
              'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'global_cap_hours':24,'free_compute_only':True,'experiment_settings_unchanged':6,
              'prior_charged_seconds':prior,'remaining_epoch_maximum_hours':remaining,
              'prior_ledger_sha256':hashlib.sha256(previous).hexdigest(),
              'accounting':'Global total = prior charged seconds + new epoch seconds. Stage totals carry forward to enforce the unchanged stage allowances. No old charge or checkpoint is discarded.'}
    receipt.write_text(json.dumps(evidence,indent=2)+'\n')
    record.update(seconds=0.,maximum_hours=remaining,global_cap_hours=24,prior_charged_seconds=prior,
                  stage_carry_seconds=dict(record['by_stage']),
                  budget_amendment_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                  measurement='New authorized epoch seconds; stage totals carry earlier charges. Add prior_charged_seconds for the global total.')
    ledger.write_text(json.dumps(record,indent=2)+'\n')
print(receipt.read_text())
