#!/bin/zsh
# Human-controlled account selection and consent; credentials never enter task artifacts.
set -u
umask 077
TASK_DIR="${0:A:h}"
PY_BIN="$TASK_DIR/../improvement_v9/colab_cli_env/bin/python"
print 'V15: connect a different Google Colab account'
print 'Choose your OTHER account in Google sign-in, then review the requested access.'
print 'Paste the authorization code into this Terminal only. Input is hidden.'
print 'Keep passwords and authorization codes out of chat.'
print 'The existing V15 eight-hour clock, sources and completed checks are preserved.'
print ''
"$PY_BIN" - "$TASK_DIR" <<'PYCODE'
import json,sys,time,os
from pathlib import Path
root=Path(sys.argv[1]);auth=json.loads((root/'authorization.json').read_text())
if time.time()-auth['started_epoch']>=8*3600:raise SystemExit('Original V15 cap expired. No new run was started.')
# Keep the old login recoverable inside the original protected credential directory.
p=Path.home()/'.config/colab-cli/token.json'
if p.exists():
    backup=p.with_name('token.before-v15-switch.'+str(int(time.time()))+'.json')
    p.rename(backup);os.chmod(backup,0o600)
(root/'transport_run/account_switch_requested.json').write_text(json.dumps({'user_authorized_different_account':True,'human_sign_in_pending':True,'original_budget_retained':True,'credentials_in_artifacts':False},indent=2)+'\n')
print('Previous CLI login preserved in its protected configuration directory.')
PYCODE
if [[ $? -ne 0 ]]; then
    read -r '?Press Return to close. '
    exit 1
fi
restore_echo() { stty echo 2>/dev/null || true; }
trap restore_echo EXIT INT TERM
stty -echo
"$PY_BIN" - "$TASK_DIR" <<'PYCODE'
import json,logging,sys,time,builtins,getpass
from pathlib import Path
logging.disable(logging.CRITICAL)
builtins.input=lambda prompt='':getpass.getpass(prompt)
from colab_cli.common import state
# Reaching account usage authenticates without allocating or uploading anything.
try:
    info=state.client.get_consumption_user_info()
except BaseException:
    print('Google sign-in was not completed. Keep codes out of chat and retry this launcher.')
    raise SystemExit(1)
root=Path(sys.argv[1]);record={'human_completed_google_sign_in':True,'user_selected_different_account':True,'original_budget_retained':True,'credentials_in_artifacts':False,'epoch':time.time()}
(root/'transport_run/account_switch_completed.json').write_text(json.dumps(record,indent=2)+'\n')
print('Google sign-in completed. Return to Codex and say: New Colab account connected.')
PYCODE
restore_echo
print ''
read -r '?Press Return to close. '
