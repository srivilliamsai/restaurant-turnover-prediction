import sys
from pathlib import Path
root=Path('./improvement_v10')
sys.path.insert(0,str(root))
from core import Engine
import build_delivery
e=Engine(root.parent,root/"colab_results/results")
build_delivery.finalize_receipt(e,root/"colab_results/delivery")
