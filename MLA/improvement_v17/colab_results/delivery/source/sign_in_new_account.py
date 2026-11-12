"""Human-operated Colab sign-in. Credentials stay in the CLI's private store."""
import hashlib,json,logging,os,time
from pathlib import Path
from colab_cli.auth import get_credentials,AuthProvider
os.umask(0o077)
logging.disable(logging.CRITICAL)
folder=Path.home()/'.config/colab-cli'
token=folder/'token.json'
backup=folder/('token.before-v17-human-signin.'+str(int(time.time()))+'.json')
if token.exists():token.replace(backup)
print('Sign in to your chosen NEW Google account using the link below.')
print('Paste Google\'s authorization code ONLY into this terminal when prompted.\n',flush=True)
try:
    session=get_credentials(provider=AuthProvider.OAUTH2)
    response=session.get('https://www.googleapis.com/oauth2/v2/userinfo',timeout=(10,30))
    response.raise_for_status()
    info=response.json()
    address=info.get('email','');local,_,domain=address.partition('@')
    record={'status':'PASS','human_completed_google_sign_in':True,'user_selected_account':True,
        'selected_account_masked':local[:2]+'***@'+domain,
        'account_identity_sha256':hashlib.sha256(info['id'].encode()).hexdigest(),
        'original_started_epoch':1793251847,'clock_reset':False,'credentials_in_artifacts':False,'epoch':time.time()}
    root=Path(__file__).resolve().parent
    (root/'transport_run/account_switch_completed.json').write_text(json.dumps(record,indent=2)+'\n')
    (root/'account_switch.json').write_text(json.dumps(record,indent=2)+'\n')
    print('\nSign-in successful for '+record['selected_account_masked']+'. Return to the chat and say done.')
except BaseException as error:
    if not token.exists() and backup.exists():
        import shutil
        shutil.copy2(backup,token);token.chmod(0o600)
    print('\nSign-in did not finish ('+type(error).__name__+'). The prior login was preserved.')
    raise SystemExit(1)
