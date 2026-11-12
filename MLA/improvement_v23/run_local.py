"""Run the finite local experiment, keeping network downloads separate."""
import json,os,signal,subprocess,sys,time
import psutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ENV=os.environ.copy()
ENV.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ENV['PYTHONPATH']=str(ROOT.parent/'.runtime_deps')+':'+str(ROOT/'runtime_deps')
AUTH=json.loads((ROOT/'authorization.json').read_text())
def run(stage):
    ceiling=AUTH['engineering_stop_epoch'] if stage in ['resolve','pilot'] else AUTH['training_stop_epoch']
    remaining=ceiling-time.time()
    if remaining<=0:raise TimeoutError('V23 allowance reached before '+stage)
    with (ROOT/'local_run.log').open('a',buffering=1) as log:
        p=subprocess.Popen([sys.executable,str(ROOT/'experiment.py'),stage],env=ENV,cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                if time.time()>=ceiling:raise TimeoutError('V23 '+stage+' elapsed ceiling reached')
                try:
                    if psutil.Process(p.pid).memory_info().rss>5*1024**3:
                        raise MemoryError('V23 local process exceeded the 5 GiB resident-memory ceiling')
                except psutil.NoSuchProcess:pass
                time.sleep(2)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    assert p.returncode==0,'V23 '+stage+' failed; inspect measured evidence'
    print(json.dumps({'stage_complete':stage,'epoch':time.time()}),flush=True)
def main():
    try:
        for stage in ['resolve','pilot','screen']:run(stage)
        if (ROOT/'results/candidate_lock.json').exists():run('confirm')
    except Exception as error:
        path=ROOT/'results/coordinator_failure.json';path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({'stage':'coordinator','type':type(error).__name__,'message':str(error),'epoch':time.time()},indent=2)+'\n')
        print(json.dumps({'execution_stopped':True,'reason':str(error)}),flush=True)
    finally:
        subprocess.run([sys.executable,str(ROOT/'finalize.py')],env=ENV,cwd=ROOT,check=True,timeout=120)
if __name__=='__main__':main()
