"""Read-only preflight inside the existing, idle V9 Colab kernel."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

v9_expected_inputs = {
    "Train_dataset.csv": "ec40ebda74ba0bfbfe1af4417a4266a70eadf6e595500e99e5435e7bdc7ee51f",
    "Test_dataset.csv": "ee4137566adb1c0ce356a8c370e7b8a22f681bde492068efe11346d8d44b4f4d",
    "Data_Dictionary.csv": "2d86c00e27448593564a161197b6639ee1eb90c01fa523b67e7a09d2ef405c5e",
    "Sample_Solution.csv": "9519ba4d51eb2d59b8a71b054a62e1f4d48b72cf92930c31f5e5393a2cd135fb",
}
v9_actual_inputs = {name: hashlib.sha256((Path('/content') / name).read_bytes()).hexdigest()
                    for name in v9_expected_inputs}
assert v9_actual_inputs == v9_expected_inputs, 'Official input hashes do not match.'
v9_versions = {}
for v9_package in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'catboost',
                   'interpret-core', 'tabpfn', 'torch']:
    try:
        v9_versions[v9_package] = importlib.metadata.version(v9_package)
    except importlib.metadata.PackageNotFoundError:
        v9_versions[v9_package] = None
print(json.dumps({'input_hashes_verified': True, 'versions': v9_versions,
                  'existing_research_root': str(globals().get('WORK_ROOT', 'not initialized'))}, indent=2))
