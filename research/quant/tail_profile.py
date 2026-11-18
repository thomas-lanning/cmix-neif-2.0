"""Capture the native operands for training the final MLP against token loss."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from evaluate import REFERENCE_SHA, model_id

ROW = np.dtype([("up", "u1", 192), ("down", "u1", 1024), ("partial", "<f4", 192),
                ("sum", "<f4", 192), ("max_logit", "<f4"), ("normalizer", "<f4")])


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def instrument(root):
    file = root/"src/tf1/opt/model_opt.cpp"
    text = file.read_text()
    anchor = "      mlp_block(l, r, n);"
    if text.count(anchor) != 1:
        raise ValueError("release MLP changed")
    block = r"""
      if (l == NL-1 && !kv_last) {
        static FILE* capture = std::fopen("tail.bin", "wb");
        if (!capture || n != 1 || M.mlp[l].down.d_in != 1024 || r->ar_n < 2 || !r->ar_has_part)
          std::exit(6);
        auto write = [&](const void* p, size_t bytes) {
          if (std::fwrite(p, 1, bytes, capture) != bytes) std::exit(6);
        };
        write(r->q5, D);
        write(r->q768, 1024);
        write(r->ar_part, sizeof(float)*D);
        write(r->ar_qA[2*NL], sizeof(float)*D);
        write(&r->ar_qM[2*NL], sizeof(float));
        write(&r->ar_qZ[2*NL], sizeof(float));
      }
"""
    file.write_text("#include <cstdio>\n#include <cstdlib>\n"+text.replace(anchor, anchor+block))


def collect(exe, suite, model, out):
    manifest = suite/"manifest.json"
    identity, fnv, _ = model_id(model)
    if identity != REFERENCE_SHA:
        raise ValueError("capture the published model")
    out.mkdir(parents=True,exist_ok=False)
    records = []
    for w in json.loads(manifest.read_bytes())["windows"]:
        source = suite/w["file"]
        if sha(source) != w["sha256"]:
            raise ValueError("fixture differs")
        part = out/source.stem
        part.mkdir()
        native = json.loads(subprocess.check_output([exe, model, fnv, source],cwd=part,text=True,timeout=300))
        f = part/"tail.bin"
        if native["targets"] != w["targets"] or f.stat().st_size != native["targets"]*ROW.itemsize:
            raise ValueError("tail capture and native target counts differ")
        records.append(dict(file=str(f.relative_to(out)),sha256=sha(f),document=w["document"],targets=w["targets"]))
    (out/"manifest.json").write_text(json.dumps(dict(suite=sha(manifest),model=identity,executable=sha(exe),
                                                     files=records),indent=2)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("instrument").add_argument("root",type=Path)
    c=sub.add_parser("collect")
    for name in ("exe","suite","model","out"):
        c.add_argument(name,type=Path)
    a=p.parse_args()
    if a.command=="instrument": instrument(a.root)
    else: collect(*(v.resolve() for v in (a.exe,a.suite,a.model,a.out)))
