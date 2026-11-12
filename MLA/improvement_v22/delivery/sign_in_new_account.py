"""Human-operated Google sign-in; no credentials enter chat or project files."""
import builtins
import getpass
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from colab_cli.auth import AuthProvider, get_credentials

os.umask(0o077)
logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parent
TOKEN = Path.home() / '.config/colab-cli/token.json'
BACKUP = TOKEN.with_name('token.before-v22-human-signin.' + str(int(time.time())) + '.json')


def main():
    released = json.loads((ROOT/'account_release_check.json').read_text())
    assert released['status'] == 'PASS' and released['live_active_assignments'] == 0
    print('V22: sign in to the Google account you want to use for Colab.')
    print('Open the Google link below. Paste the authorization code ONLY into this terminal.')
    print('The code is hidden while pasted. Press Return to finish. No GPU is allocated here.\n', flush=True)
    if TOKEN.exists():
        shutil.copy2(TOKEN, BACKUP)
        BACKUP.chmod(0o600)
        TOKEN.unlink()
    original_input = builtins.input
    builtins.input = lambda prompt='': getpass.getpass(prompt + ' (hidden input) ')
    try:
        session = get_credentials(provider=AuthProvider.OAUTH2)
        response = session.get('https://www.googleapis.com/oauth2/v2/userinfo', timeout=(10,30))
        response.raise_for_status()
        info = response.json()
        identity = hashlib.sha256(info['id'].encode()).hexdigest()
        TOKEN.chmod(0o600)
        local, _, domain = info.get('email','').partition('@')
        record = {'status':'PASS','human_completed_google_sign_in':True,
                  'user_selected_account':True,'selected_account_masked':local[:2]+'***@'+domain,
                  'account_identity_sha256':identity,
                  'different_from_pre_switch_account':identity != released['account_identity_sha256'],
                  'credentials_in_artifacts':False,'gpu_allocated':False,
                  'original_started_epoch':json.loads((ROOT/'authorization.json').read_text())['started_epoch'],
                  'clock_reset':False,'checked_epoch':time.time()}
        (ROOT/'account_switch.json').write_text(json.dumps(record,indent=2)+'\n')
        print('\nColab sign-in verified for '+record['selected_account_masked'])
        print('Return to the chat and say done.', flush=True)
    except BaseException as error:
        if BACKUP.exists():
            shutil.copy2(BACKUP,TOKEN)
            TOKEN.chmod(0o600)
        elif TOKEN.exists(): TOKEN.unlink()
        (ROOT/'account_switch.json').write_text(json.dumps({
            'status':'INCOMPLETE','error_type':type(error).__name__,
            'previous_login_restored':BACKUP.exists(),'credentials_in_artifacts':False,
            'gpu_allocated':False,'clock_reset':False,'checked_epoch':time.time()},indent=2)+'\n')
        print('\nSign-in incomplete ('+type(error).__name__+'). Previous login preserved.')
        raise SystemExit(1)
    finally: builtins.input = original_input


if __name__ == '__main__': main()
