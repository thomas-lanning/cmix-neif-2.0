"""First post-training quantization hypothesis. Edit this file for a new idea."""
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ["NEIF_SOURCE_ROOT"]) if "NEIF_SOURCE_ROOT" in os.environ else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/tf/3_train/export/np2"))
import np2_container as np2


def export(source, output, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    tensors = np2.read_container(str(source))
    # Coarsen the lowest-error MLP down-projection rows in the final MLP.
    eligible = [t for t in tensors if t.name.endswith("mlp.down.weight.q") and t.a.dtype == np.int8]
    tensor = eligible[-1]
    rows = tensor.m2()
    # Keep the symmetric endpoints. This reduces the number of interior symbols.
    coarse = np.clip(2 * np.rint(rows.astype(np.float32) / 2), -7, 7).astype(np.int8)
    error = ((coarse.astype(np.int16)-rows)**2).sum(axis=1)
    changed = np.flatnonzero(error)
    chosen = changed[np.argsort(error[changed], kind="stable")[:max(1, int(len(changed)*fraction))]]
    if not len(chosen):
        raise ValueError("no eligible rows")
    rows[chosen] = coarse[chosen]
    groups = np2.np_groups(tensors, True, True)
    blob, _, permuted = np2.encode_np(tensors, True, groups, True)
    with output.open("xb") as f:
        f.write(blob)
    actual = np2.read_container(str(output))
    if len(actual) != len(permuted) or not all(a.same(b) for a,b in zip(actual, permuted)):
        raise ValueError("candidate failed its tensor round trip")
    return dict(tensor=tensor.name, changed_rows=len(chosen))


if __name__ == "__main__":
    print(export(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])), flush=True)
