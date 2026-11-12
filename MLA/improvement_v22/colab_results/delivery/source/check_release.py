"""Verify selected account identity and runtime release without printing credentials."""
import hashlib,json,time
from pathlib import Path
from colab_cli.common import state
ROOT=Path(__file__).resolve().parent

def main():
    r=state.client.session.get('https://www.googleapis.com/oauth2/v2/userinfo',timeout=(10,30))
    r.raise_for_status()
    identity=hashlib.sha256(r.json()['id'].encode()).hexdigest()
    expected=json.loads((ROOT/'authorization.json').read_text())['account_identity_sha256']
    assert identity==expected
    active=state.client.list_assignments()
    record={'status':'PASS' if not active else 'ACTIVE_ASSIGNMENTS_REMAIN',
            'checked_epoch':time.time(),'account_identity_sha256':identity,
            'selected_account_identity_verified':True,'live_active_assignments':len(active),
            'owned_session':'v22-free-t4'}
    (ROOT/'account_release_check.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    assert not active,'A runtime remains active; do not stop unrelated work.'

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('Release verification failed:',type(error).__name__,'; authenticated details suppressed.',flush=True)
        raise SystemExit(1)

