"""Repackage final receipts after all supervised computation is charged."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core import Engine
import build_delivery
e=Engine(ROOT.parent,ROOT/'colab_results/results');assert e.cache.state()['seconds']<=64800
decision=json.loads((e.out/'decision.json').read_text());build_delivery.package(e,decision,ROOT/'colab_results/delivery')
print('Final V12 receipts and user-approved eighteen-hour cap verified.',flush=True)
