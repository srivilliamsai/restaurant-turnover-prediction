"""Execute every notebook cell; use hash-verified actual fold caches transparently."""
from pathlib import Path
import os, json, io, contextlib, time, base64, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

ROOT=Path(__file__).resolve().parent
STEM='submission'
os.chdir(ROOT)
os.environ['MLA_VALIDATION_CACHE']=str(ROOT/'results')
path=ROOT/(STEM+'.ipynb')
book=nbformat.read(path,as_version=4)
namespace={'__name__':'__main__'}
count=0;start=time.time();figure_count=0
for index,cell in enumerate(book.cells):
    if cell.cell_type!='code':continue
    count+=1;cell.execution_count=count
    out,err=io.StringIO(),io.StringIO();figures=[]
    def show(*args,**kwargs):
        global figure_count
        for number in plt.get_fignums():
            fig=plt.figure(number);stream=io.BytesIO();fig.savefig(stream,format='png',dpi=130,bbox_inches='tight')
            figure_count+=1
            (ROOT/'results'/f'notebook_figure_{figure_count}.png').write_bytes(stream.getvalue())
            figures.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(stream.getvalue()).decode(),'text/plain':'Executed notebook figure'},metadata={}))
            plt.close(fig)
    plt.show=show
    print('Executing cell',index,'code',count,flush=True)
    try:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            exec(compile(cell.source,f'cell_{index}','exec'),namespace)
    except Exception as e:
        cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=out.getvalue()),nbformat.v4.new_output('error',ename=type(e).__name__,evalue=str(e),traceback=traceback.format_exc().splitlines())]
        nbformat.write(book,path);raise
    cell.outputs=[]
    if out.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=out.getvalue()))
    if err.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=err.getvalue()))
    cell.outputs.extend(figures);nbformat.write(book,path)
    print('Done cell',index,'elapsed',round(time.time()-start,1),flush=True)

# The notebook full fit must reproduce the independently fitted candidate.
expected=np.load(ROOT/'results/full_prediction_parts.npz')['candidate']
assert np.allclose(expected,namespace['test_prediction'],rtol=1e-12,atol=.01)
nbformat.validate(book)
summary=dict(code_cells=count,figures=figure_count,elapsed_seconds=time.time()-start,
             validation=namespace['validation_table'].to_dict(orient='records'),
             means=namespace['means'].to_dict(),bootstrap_interval=namespace['gain_ci'].tolist(),
             validation_execution='Reused hash-verified actual fold predictions; recomputed metrics; full fits and behavioral/export checks executed in this notebook.',
             full_fit_reproduced=True)
(ROOT/'results/notebook_execution.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
