import sys
from pathlib import Path

rootdir = str(Path(__file__).parent.parent)
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)
