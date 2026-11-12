"""Execute a standalone candidate notebook in a clean process, saving every cell."""
import contextlib
import io
import sys
from pathlib import Path
import nbformat as nb

path=Path(sys.argv[1]);book=nb.read(path,as_version=4)
namespace={'__name__':'__main__'}
sys.modules['google.colab']=None
class Tee(io.StringIO):
    def write(self,value):
        sys.__stdout__.write(value);sys.__stdout__.flush()
        return super().write(value)
for count,cell in enumerate([c for c in book.cells if c.cell_type=='code'],1):
    capture=Tee();cell.execution_count=count
    try:
        with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
            exec(compile(cell.source,'<portable standalone notebook>','exec'),namespace)
    except Exception:
        cell.outputs=[nb.v4.new_output('stream',name='stdout',text=capture.getvalue())]
        nb.write(book,path)
        raise
    cell.outputs=[nb.v4.new_output('stream',name='stdout',text=capture.getvalue())]
    nb.write(book,path)
print('FRESH_NOTEBOOK_COMPLETE',count,flush=True)
