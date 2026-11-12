#!/bin/zsh
# Human-controlled Google sign-in. Existing V12 budget/checkpoints are preserved.
set -u
umask 077
TASK_DIR="${0:A:h}"
COLAB_BIN="$TASK_DIR/../improvement_v9/colab_cli_env/bin/colab"
PY_BIN="$TASK_DIR/../improvement_v9/colab_cli_env/bin/python"

print 'Switch the Google account used by the Colab CLI'
print ''
print '1. Open the Google sign-in URL printed below (Command-click it).'
print '2. Choose Use another account and sign in with your different Colab account.'
print '3. Review Google access permissions and approve them if you agree.'
print '4. Paste the authorization code into THIS Terminal window and press Return.'
print '   Typed/pasted code will be hidden. Keep codes and passwords out of chat.'
print ''
print 'Your three passed pilot checks and all previous charges are preserved.'
print 'The user-approved total cap is 18 hours. No paid compute will be used.'
print 'This step signs in only. Return to Codex after it finishes.'
print ''

if ! "$PY_BIN" - "$TASK_DIR" <<'PY'
import json,os,sys
from pathlib import Path
root=Path(sys.argv[1]);path=root/'transport_run/coordinator_pid.json'
if path.exists():
    pid=json.loads(path.read_text())['pid']
    try:os.kill(pid,0)
    except ProcessLookupError:pass
    else:raise SystemExit('V12 coordinator is still running. Account switch stopped to protect its session.')
token=Path.home()/'.config/colab-cli/token.json'
token.unlink(missing_ok=True)
print('Previous local CLI login cleared. Google account data and model files are unchanged.')
PY
then
    read -r '?Press Return to close. '
    exit 1
fi

restore_echo() { stty echo 2>/dev/null || true; }
trap restore_echo EXIT INT TERM
stty -echo
if "$COLAB_BIN" --auth oauth2 sessions; then
    restore_echo
    print ''
    print 'Google sign-in completed with the account you selected.'
    print ''
    print 'Return to Codex and say: New Colab account connected.'
    print 'V12 will resume after the disconnect-recovery checks are ready.'
else
    restore_echo
    print ''
    print 'Sign-in did not complete. Tell Codex the error text without codes or tokens.'
fi
print ''
read -r '?Press Return to close this window. '
