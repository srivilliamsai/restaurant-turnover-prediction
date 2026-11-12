"""Finite V22 coordinator; keeps foreground computation connected and backs up folds."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONTROL=ROOT/'colab_control.py'
CLI=ROOT.parent/'improvement_v9/colab_cli_env/bin/colab'
RUN=ROOT/'transport_run'
RUN.mkdir(exist_ok=True)
AUTH=json.loads((ROOT/'authorization.json').read_text())
ENV=os.environ.copy()
ENV['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:
    ENV[key]='2'
ENV['PYTHONPATH']=str(ROOT.parent/'.runtime_deps')+':'+str(ROOT.parent/'improvement_v9/runtime_deps')+':'+str(ROOT)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)
def status(phase,**extra):
    x={'phase':phase,'pid':os.getpid(),'task_elapsed_seconds':time.time()-AUTH['started_epoch'],'maximum_hours':4,**extra}
    save(RUN/'status.json',x)
    print(json.dumps(x),flush=True)
def remaining():
    return max(0.,4*3600-(time.time()-AUTH['started_epoch']))
def stop(process):
    if process is None or process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
def command(arguments,timeout,check=True,backup=False):
    process=subprocess.Popen([str(x) for x in arguments],stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=subprocess.STDOUT,env=ENV,cwd=ROOT.parent,start_new_session=True)
    deadline=time.monotonic()+min(timeout,remaining())
    last_backup=time.monotonic()
    failures=0
    try:
        while process.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('V22 coordinator allowance exhausted')
            if backup and time.monotonic()-last_backup>=60:
                try:
                    result=subprocess.run([sys.executable,str(CONTROL),'snapshot'],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=120,env=ENV)
                    print(result.stdout,end='',flush=True)
                    failures=0 if result.returncode==0 else failures+1
                except subprocess.TimeoutExpired:
                    failures+=1
                    print(json.dumps({'backup_timeout':True,'consecutive_failures':failures}),flush=True)
                last_backup=time.monotonic()
                heartbeat=ROOT/'colab_results/results/runtime_heartbeat.json'
                if heartbeat.exists():
                    x=json.loads(heartbeat.read_text())
                    if x.get('status')=='running' and time.time()-x['epoch']>300:raise RuntimeError('V22 remote heartbeat stale; completed backups preserved')
                if failures>=3:raise RuntimeError('V22 backup transport unavailable after three attempts')
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
    except BaseException:
        stop(process)
        raise
    if check and process.returncode:raise RuntimeError('V22 command failed; inspect stage status and run.log')
    return process.returncode
def control(action,value=None,timeout=180,backup=False):
    return command([sys.executable,CONTROL,action,*([value] if value else [])],timeout,backup=backup)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',action='store_true',help='Existing owned T4 already has pinned dependencies installed.')
    parser.add_argument('--resume-pilot',action='store_true',help='Reattach to the unchanged running pilot after a local transport failure.')
    args=parser.parse_args()
    released=False
    completed=False
    assertion=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) if shutil.which('caffeinate') else None
    try:
        if not args.prepared and not args.resume_pilot:
            status('checking_user_selected_account')
            control('account-check',timeout=120)
            allocation_path=RUN/'allocations.json'
            history=json.loads(allocation_path.read_text()) if allocation_path.exists() else []
            assert len(history)<3,'Initial allocation plus two bounded recoveries exhausted'
            history.append({'requested_epoch':time.time(),'allocation_number':len(history)+1,'same_user_selected_account':True})
            save(allocation_path,history)
            status('allocating_owned_free_t4',allocation_number=len(history))
            control('allocate',timeout=240)
            status('installing_dependencies')
            control('setup',timeout=1700)
        if args.resume_pilot:
            status('reattaching_running_pilot')
            failures=0
            while time.time()<AUTH['training_stop_epoch']:
                try:
                    control('snapshot',timeout=150)
                    failures=0
                except (RuntimeError,subprocess.TimeoutExpired):
                    failures+=1
                    assert failures<3,'Pilot reattachment backups failed three times'
                    time.sleep(20)
                    continue
                out=ROOT/'colab_results/results'
                audit=json.loads((out/'audit.json').read_text())
                frozen=json.loads((ROOT/'source_frozen.json').read_text())
                assert audit['sources']==frozen['sources'] and audit['inputs']==frozen['inputs']
                progress=json.loads((out/'stage_status.json').read_text())
                assert progress.get('status')!='failed','Remote pilot failed'
                if progress=={'stage':'pilot','status':'complete'}:break
                time.sleep(30)
            else:raise TimeoutError('Original pilot allowance exhausted during transport recovery')
        else:
            status('restarting_clean_kernel')
            command([CLI,'--auth','oauth2','restart-kernel','-s','v22-rank-check-t4'],120)
            status('uploading_frozen_workflow')
            upload_ok=False
            for attempt in range(2):
                try:
                    control('upload',timeout=600)
                    upload_ok=True
                    break
                except RuntimeError:
                    status('upload_retry',attempt=attempt+1)
            assert upload_ok,'V22 upload failed after two bounded attempts'
            for stage in ['audit','pilot']:
                status('running_'+stage)
                control('stage',stage,timeout=remaining(),backup=True)
        for stage in ['screen']:
            status('running_'+stage)
            control('stage',stage,timeout=remaining(),backup=True)
        control('snapshot',timeout=180)
        lock=json.loads((ROOT/'colab_results/results/candidate_lock.json').read_text())
        if lock['choices']:
            status('using_hash_verified_original_confirmation_controls')
        for stage in ['confirm','export']:
            status('running_'+stage)
            control('stage',stage,timeout=remaining(),backup=True)
        control('snapshot',timeout=180)
        status('collecting_completed_artifacts')
        control('collect',timeout=600)
        completed=True
    except Exception as error:
        status('stopped',error_type=type(error).__name__,reason=str(error)[:300])
        try:control('snapshot',timeout=180)
        except Exception:pass
        try:control('collect',timeout=600)
        except Exception:pass
        raise
    finally:
        try:
            # Release is mandatory even if all experiment time has elapsed.
            p=subprocess.run([sys.executable,str(CONTROL),'release'],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=150,env=ENV)
            print(p.stdout,end='',flush=True)
            released=p.returncode==0
        except Exception:released=False
        save(RUN/'runtime_release.json',{'owned_session':'v22-rank-check-t4','release_command_succeeded':released,'epoch':time.time()})
        if assertion is not None:
            assertion.terminate()
            assertion.wait(timeout=5)
        status('complete' if completed else 'stopped',artifacts_collected=completed,owned_runtime_released=released)
if __name__=='__main__':main()



