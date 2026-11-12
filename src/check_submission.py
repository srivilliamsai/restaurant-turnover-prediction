"""Verify the delivered files and the prediction CSV using the standard library."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'artifacts/manifest.json'
ID = 'Registration Number'
TARGET = 'Annual Turnover'


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_hashes(directory, entries):
    for name, expected in entries.items():
        if sha256(directory / name) != expected:
            raise ValueError(f'File checksum mismatch: {name}')


def check_predictions(path, data):
    with (data / 'Test_dataset.csv').open(newline='', encoding='utf-8') as handle:
        expected_ids = [row[ID] for row in csv.DictReader(handle)]
    if len(expected_ids) != 500 or len(set(expected_ids)) != 500:
        raise ValueError('Expected 500 unique restaurant IDs in the test dataset')

    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle, strict=True)
        if next(reader, None) != [ID, TARGET]:
            raise ValueError(f'Expected CSV columns {ID!r} and {TARGET!r} in order')
        for row_number, (row, expected_id) in enumerate(zip(reader, expected_ids, strict=True), 2):
            if len(row) != 2 or row[0] != expected_id:
                raise ValueError(f'Invalid columns or restaurant ID on line {row_number}')
            turnover = float(row[1])
            if not math.isfinite(turnover) or turnover < 0:
                raise ValueError(f'Invalid predicted turnover on line {row_number}')
    return len(expected_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, nargs='?', default=ROOT / 'submission.csv')
    args = parser.parse_args()
    try:
        manifest = json.loads(MANIFEST.read_text())
        verify_hashes(ROOT / 'data', manifest['inputs'])
        verify_hashes(ROOT, manifest['artifacts'])
        rows = check_predictions(args.csv, ROOT / 'data')
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f'Verification failed: {error}\n')
    print(f'PASS: input and artifact checksums, CSV schema and {rows} ordered restaurant predictions.')


if __name__ == '__main__':
    main()
