"""Record installed versions without credentials or repository URLs."""
import importlib.metadata
import json
import platform
from pathlib import Path
import torch

v9_environment = {
    'python': platform.python_version(),
    'platform': platform.platform(),
    'packages': dict(sorted({d.metadata['Name']: d.version
                             for d in importlib.metadata.distributions()
                             if d.metadata.get('Name')}.items(), key=lambda x: x[0].lower())),
    'torch_cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'checkpoint': json.loads((WORK_ROOT / 'tabpfn_checkpoint.json').read_text()),
}
v9_environment_path = OUTPUT_DIR / 'colab_environment_manifest.json'
v9_environment_path.write_text(json.dumps(v9_environment, indent=2) + '\n')
print('Environment manifest saved:', v9_environment_path)
