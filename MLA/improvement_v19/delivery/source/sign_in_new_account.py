"""User-operated Colab sign-in; authorization codes never enter chat or logs."""
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
BACKUP = TOKEN.with_name('token.before-v19-human-signin.' + str(int(time.time())) + '.json')
PREVIOUS = json.loads((ROOT.parent / 'improvement_v18/account_switch.json').read_text())


def private_input(prompt=''):
    return getpass.getpass(prompt + ' (hidden input) ')


def main():
    print('V19: sign in to the NEW Google account you want to use for Colab.')
    print('Open the Google link below; paste its authorization code ONLY here.')
    print('No GPU will be allocated by this sign-in step.\n', flush=True)
    if TOKEN.exists():
        shutil.copy2(TOKEN, BACKUP)
        BACKUP.chmod(0o600)
        TOKEN.unlink()
    previous_input = builtins.input
    builtins.input = private_input
    try:
        session = get_credentials(provider=AuthProvider.OAUTH2)
        response = session.get('https://www.googleapis.com/oauth2/v2/userinfo', timeout=(10, 30))
        response.raise_for_status()
        info = response.json()
        identity = hashlib.sha256(info['id'].encode()).hexdigest()
        if identity == PREVIOUS['account_identity_sha256']:
            raise ValueError('same_account')
        TOKEN.chmod(0o600)
        local, _, domain = info.get('email', '').partition('@')
        record = {
            'status': 'PASS',
            'human_completed_google_sign_in': True,
            'user_selected_account': True,
            'different_from_v18': True,
            'selected_account_masked': local[:2] + '***@' + domain,
            'account_identity_sha256': identity,
            'credentials_in_artifacts': False,
            'gpu_allocated': False,
            'training_started': False,
            'checked_epoch': time.time(),
        }
        (ROOT / 'account_switch.json').write_text(json.dumps(record, indent=2) + '\n')
        print('\nNew Colab account verified: ' + record['selected_account_masked'])
        print('Sign-in complete. Return to the chat and say done.', flush=True)
    except BaseException as error:
        if BACKUP.exists():
            shutil.copy2(BACKUP, TOKEN)
            TOKEN.chmod(0o600)
        elif TOKEN.exists():
            TOKEN.unlink()
        reason = 'The previous account was selected.' if isinstance(error, ValueError) and str(error) == 'same_account' else type(error).__name__
        (ROOT / 'account_switch.json').write_text(json.dumps({
            'status': 'INCOMPLETE', 'reason': reason,
            'previous_login_restored': BACKUP.exists(),
            'credentials_in_artifacts': False, 'gpu_allocated': False,
            'checked_epoch': time.time(),
        }, indent=2) + '\n')
        print('\nSign-in incomplete (' + reason + '). Previous login preserved.', flush=True)
        raise SystemExit(1)
    finally:
        builtins.input = previous_input


if __name__ == '__main__':
    main()
