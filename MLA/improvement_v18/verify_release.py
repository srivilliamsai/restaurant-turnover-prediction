"""Verify current-account release without stopping unrelated assignments."""
import hashlib,json,logging,os,time
from pathlib import Path
os.environ['COLAB_CLI_REQUEST_TIMEOUT']='30'
logging.disable(logging.CRITICAL)
from colab_cli.common import state
ROOT=Path(__file__).resolve().parent
release=json.loads((ROOT/'transport_run/runtime_release.json').read_text())
assert release['owned_session']=='v18-free-t4' and release['release_command_succeeded']
try:
 response=state.client.session.get('https://www.googleapis.com/oauth2/v2/userinfo',timeout=(10,30))
 response.raise_for_status()
 identity=hashlib.sha256(response.json()['id'].encode()).hexdigest()
 switch=json.loads((ROOT/'account_switch.json').read_text())
 assert identity==switch['account_identity_sha256'],'Current account differs from the user-selected account'
 assignments=state.client.list_assignments()
 record={'status':'PASS' if len(assignments)==0 else 'CHECK_REQUIRED',
  'method':'Authenticated fresh Colab list_assignments response; no credentials recorded',
  'live_active_assignments':len(assignments),'checked_epoch':time.time(),
  'owned_session':'v18-free-t4','owned_session_release_command_succeeded':True,
  'user_selected_account_verified':True,'account_identity_sha256':identity,
  'unrelated_assignments_stopped':False,'paid_resources':False}
 (ROOT/'account_release_check.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record,indent=2))
except Exception as error:
 print(json.dumps({'status':'CHECK_FAILED','error_type':type(error).__name__}))
 raise SystemExit(1)
