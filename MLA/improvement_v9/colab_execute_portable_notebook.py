"""Execute the draft standalone notebook; keep release qualification separate."""
import ast
import hashlib
import json
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime
root=Path(__file__).resolve().parent
run=root/'colab_cli_runs/20261026T160148Z'
draft=run/'recovery_draft'
remote='/content/v9_cli_20261026T160148Z'
session=state.get_session('v9-cli-t4')
contents=ContentsClient(session)
# Two named, user-owned source artifacts only; no directory upload.
contents.upload(str(draft/'V9_Portable_Colab.ipynb'),remote+'/V9_Portable_Colab.ipynb')
contents.upload(str(root/'portable_notebook_executor.py'),remote+'/portable_notebook_executor.py')
runtime=ColabRuntime(session.url,session.token,kernel_id=session.kernel_id,session_id=session.session_id)
try:
    source=f'''
import subprocess,os
portable_environment=os.environ.copy()
portable_environment['MLA_DATA_DIR']='/content'
portable_environment['V9_PORTABLE_OUTPUT_DIR']={remote+'/portable_fresh_execution'!r}
portable_process=subprocess.Popen([sys.executable,'-u',{remote+'/portable_notebook_executor.py'!r},{remote+'/V9_Portable_Colab.ipynb'!r}],env=portable_environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
for line in portable_process.stdout:print(line,end='',flush=True)
assert portable_process.wait()==0,'Standalone candidate notebook failed.'
'''
    outputs=runtime.execute_code(source,timeout=3600,output_hook=lambda x:print(x.get('text',''),end='',flush=True) if x.get('output_type')=='stream' else None)
    for x in outputs:
        if x.get('output_type')=='error':raise RuntimeError(x.get('ename','Error')+': '+x.get('evalue',''))
    allowlist={remote+'/V9_Portable_Colab.ipynb':draft/'V9_Portable_Colab_executed.ipynb',
               remote+'/portable_fresh_execution/fresh_execution_checks.json':draft/'fresh_execution_checks.json',
               remote+'/portable_fresh_execution/submission.csv':draft/'fresh_execution.csv'}
    for source,destination in allowlist.items():
        contents.download(source,str(destination))
        if destination.suffix=='.ipynb':
            # The official CLI currently stringifies notebook content dictionaries
            # instead of JSON-serializing them. Recover only literal data, never eval.
            import nbformat
            try: notebook_data=json.loads(destination.read_text())
            except json.JSONDecodeError:notebook_data=ast.literal_eval(destination.read_text())
            recovered=nbformat.from_dict(notebook_data)
            nbformat.validate(recovered);nbformat.write(recovered,destination)
    print('Fresh standalone execution downloaded; qualification still required.',flush=True)
finally:runtime.stop()
