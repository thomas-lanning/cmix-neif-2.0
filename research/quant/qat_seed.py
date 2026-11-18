"""Recover the published weights in the format consumed by container_to_tch.py."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from evaluate import BASELINE, REFERENCE_SHA, ROOT

sys.path.insert(0, str(ROOT / "training/tf/3_train/export/np2"))
import np2_container as np2


def make_seed(destination):
    if hashlib.sha256(BASELINE.read_bytes()).hexdigest() != REFERENCE_SHA:
        raise ValueError("release container digest differs")
    original = np2.read_container(str(BASELINE))
    # The trainer reads plain v5. Its dictionary keys are the tensor names.
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        plain = Path(tmp) / "release.c5"
        np2.wc.write_tensor_file_v5(str(plain), np2.entries_of(original), k=True)
        restored = np2.wc.read_tensor_file_v3(str(plain))
        if len(restored) != len(original):
            raise ValueError("tensor count changed")
        for tensor in original:
            code, values = restored[tensor.name]
            if not tensor.same(np2.Ent(tensor.name, code, values)):
                raise ValueError(f"plain v5 does not reproduce {tensor.name}")
        result = dict(published_np2_sha256=REFERENCE_SHA, tensors=len(original),
                      tensor_digest=np2.tensors_sha(original), plain_v5_bytes=plain.stat().st_size,
                      plain_v5_sha256=hashlib.sha256(plain.read_bytes()).hexdigest())
        destination.hardlink_to(plain)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_v5", type=Path)
    args = parser.parse_args()
    print(json.dumps(make_seed(args.new_v5), indent=2))
