"""Materialize immutable V14 controls and legacy code without changing their bytes."""
import hashlib
import json
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'improvement_v15_retry1'

def main():
    for folder in ['reference', 'provenance']:
        shutil.copytree(PREVIOUS / folder, ROOT / folder, dirs_exist_ok=True)
    for name in ['baseline.py', 'legacy_models.py', 'legacy_reference.py', 'frozen_v9_modeling.py',
                 'v5_contract.json', 'city_map.json', 'reference_manifest.json', 'baseline_registry.json']:
        shutil.copy2(PREVIOUS / name, ROOT / name)
    records = {}
    for folder in ['reference', 'provenance']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file():
                source = PREVIOUS / path.relative_to(ROOT)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
                records[str(path.relative_to(ROOT))] = digest
    (ROOT / 'inherited_control_copy.json').write_text(json.dumps({'status': 'PASS', 'files': records}, indent=2) + '\n')
    print('Immutable V14 controls copied and hash-verified:', len(records))

if __name__ == '__main__':
    main()

