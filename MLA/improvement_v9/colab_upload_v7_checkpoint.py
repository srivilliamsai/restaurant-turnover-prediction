"""Transfer one verified, user-owned V7 model to the existing V9 runtime."""
import hashlib
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient

root=Path(__file__).resolve().parent
p=root/'colab_cli_runs/20261026T160148Z/export_diagnostics/local_v7_fit.pkl'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='63ab212c7249ab857fc52004a97341662c5b110db7e21f19ae6b20dcdfc1090d'
remote='/content/v9_cli_20261026T160148Z/v9_fresh__h2_j7gd/export_diagnostics/local_v7_fit.pkl'
ContentsClient(state.get_session('v9-cli-t4')).upload(str(p),remote)
print('Uploaded the single verified V7 model checkpoint; no portal upload.')
