"""Finite V16 coordinator; keeps foreground computation connected and backs up folds."""
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
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)
def status(phase,**extra):
    x={'phase':phase,'pid':os.getpid(),'task_elapsed_seconds':time.time()-AUTH['started_epoch'],'maximum_hours':6,**extra}
    save(RUN/'status.json',x)
    print(json.dumps(x),flush=True)
def remaining():
    return max(0.,6*3600-(time.time()-AUTH['started_epoch']))
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
            if time.monotonic()>deadline:raise TimeoutError('V16 coordinator allowance exhausted')
            if backup and time.monotonic()-last_backup>=60:
                result=subprocess.run([sys.executable,str(CONTROL),'snapshot'],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=90,env=ENV)
                print(result.stdout,end='',flush=True)
                failures=0 if result.returncode==0 else failures+1
                last_backup=time.monotonic()
                heartbeat=ROOT/'colab_results/results/runtime_heartbeat.json'
                if heartbeat.exists():
                    x=json.loads(heartbeat.read_text())
                    if x.get('status')=='running' and time.time()-x['epoch']>300:raise RuntimeError('V16 remote heartbeat stale; completed backups preserved')
                if failures>=3:raise RuntimeError('V16 backup transport unavailable after three attempts')
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
    except BaseException:
        stop(process)
        raise
    if check and process.returncode:raise RuntimeError('V16 command failed; inspect stage status and run.log')
    return process.returncode
def control(action,value=None,timeout=180,backup=False):
    return command([sys.executable,CONTROL,action,*([value] if value else [])],timeout,backup=backup)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',action='store_true',help='Existing owned T4 already has pinned dependencies installed.')
    args=parser.parse_args()
    released=False
    completed=False
    assertion=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) if shutil.which('caffeinate') else None
    try:
        if not args.prepared:
            status('installing_dependencies')
            control('setup',timeout=1700)
        status('restarting_clean_kernel')
        command([CLI,'--auth','oauth2','restart-kernel','-s','v16-free-t4'],120)
        status('uploading_frozen_workflow')
        control('upload',timeout=600)
        status('running_fixed_pipeline')
        control('pipeline',timeout=21600,backup=True)
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
        save(RUN/'runtime_release.json',{'owned_session':'v16-free-t4','release_command_succeeded':released,'epoch':time.time()})
        if assertion is not None:
            assertion.terminate()
            assertion.wait(timeout=5)
        status('complete' if completed else 'stopped',artifacts_collected=completed,owned_runtime_released=released)
if __name__=='__main__':main()


