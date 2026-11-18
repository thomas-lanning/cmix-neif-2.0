"""Preserve effective table weights while improving their signed representation."""
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    root = Path(os.environ["NEIF_SOURCE_ROOT"]).resolve()
    sys.path.insert(0, str(root / "research" / "quant"))
    from symmetry import BASELINE, export_one

    source, destination, threshold = sys.argv[1:]
    if Path(source).resolve() != BASELINE.resolve():
        raise ValueError("this exporter requires the released model")
    export_one("tables", "fit", float(threshold), Path(destination))
