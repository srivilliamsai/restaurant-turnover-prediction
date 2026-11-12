"""Conservative accounting for the interrupted, unobserved free runtime."""
import datetime
import json
import time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'colab_results/results'
ledger=out/'compute_budget.json'
record=json.loads(ledger.read_text())
evidence=out/'runtime_interruption.json'
if not evidence.exists():
    last=(root/'colab_results/run.log').stat().st_mtime
    gap=max(0,time.time()-last)
    details={'status':'runtime_expired_no_active_server_assignment',
             'last_backup_utc':datetime.datetime.fromtimestamp(last,datetime.timezone.utc).isoformat(),
             'recovery_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'no_completed_v10_screening_folds_in_backup':True,
             'unobserved_elapsed_seconds_charged_as_upper_bound':gap,
             'interpretation':'Actual GPU computation during this gap is unknown. Charge the full unobserved interval conservatively against the total 18-hour allowance, including possible idle/disconnected time.',
             'pilot_retry_monitored_seconds':3100}
    record['seconds']+=gap
    record['by_stage']['unobserved_runtime_interruption']=gap
    evidence.write_text(json.dumps(details,indent=2)+'\n')
    ledger.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'cumulative_charged_hours':record['seconds']/3600,
                  'remaining_total_hours':18-record['seconds']/3600}))
