"""Quarantine orphaned research checkpoints before starting a stopped workflow."""
import argparse
from pathlib import Path


def prepare(root):
    cache = Path(root) / 'results/cache'
    recovered = []
    if cache.exists():
        for path in cache.glob('*.npz'):
            if '.incomplete' in path.stem:
                continue
            if not path.with_suffix('.json').exists():
                destination = path.with_suffix('.incomplete.npz')
                number = 1
                while destination.exists():
                    destination = path.with_suffix(f'.incomplete{number}.npz')
                    number += 1
                path.rename(destination)
                recovered.append(str(destination))
    return recovered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print('Orphan checkpoints preserved for recovery:', prepare(args.root))
