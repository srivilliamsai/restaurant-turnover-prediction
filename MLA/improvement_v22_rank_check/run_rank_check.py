"""Finite supplementary coordinator; same original clock, one owned free T4."""
import json
import subprocess
import sys
import time
from run_colab import ROOT, AUTH, CLI, CONTROL, ENV, RUN, save, status, remaining, command, control


def main():
    completed=False;allocated=False
    try:
        status('checking_selected_account')
        control('account-check',timeout=120)
        status('allocating_one_owned_free_t4')
        control('allocate',timeout=240);allocated=True
        status('installing_pinned_environment')
        control('setup',timeout=1500)
        command([CLI,'--auth','oauth2','restart-kernel','-s','v22-rank-check-t4'],120)
        control('upload',timeout=600)
        for stage in ['confirm','export']:
            status('exploratory_'+stage)
            control('stage',stage,timeout=remaining(),backup=True)
            control('snapshot',timeout=180)
        control('collect',timeout=600)
        completed=True
    except Exception as error:
        status('stopped',reason=type(error).__name__)
        if allocated:
            try:control('snapshot',timeout=180)
            except Exception:pass
            try:control('collect',timeout=600)
            except Exception:pass
        raise
    finally:
        released=not allocated
        if allocated:
            try:
                p=subprocess.run([sys.executable,str(CONTROL),'release'],stdin=subprocess.DEVNULL,
                    capture_output=True,text=True,timeout=150,env=ENV)
                print(p.stdout,end='',flush=True);released=p.returncode==0
            except Exception:released=False
        save(RUN/'runtime_release.json',{'owned_session':'v22-rank-check-t4','release_command_succeeded':released,'epoch':time.time()})
        status('complete' if completed else 'stopped',artifacts_collected=completed,owned_runtime_released=released)


if __name__=='__main__':main()
