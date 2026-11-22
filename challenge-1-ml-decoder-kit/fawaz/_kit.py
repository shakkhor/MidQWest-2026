"""Make the kit's modules (circuit.py, baseline.py, decoders.py) importable from this subfolder.

The kit folder is APPENDED to the search path, so files in this folder (e.g. our tcbp.py) are
always found first, even if a teammate adds a file with the same name to the kit folder.
"""
import sys
from pathlib import Path

KIT = str(Path(__file__).resolve().parent.parent)
if KIT not in sys.path:
    sys.path.append(KIT)
