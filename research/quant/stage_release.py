"""Stage a frozen NP2 candidate in a fresh tree using the published release build."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
import tempfile

from candidate import np2
from evaluate import BASELINE, REFERENCE_SHA, ROOT, model_id

UPSTREAM = "35d1957a5074db802fc134126afc77b76d931761"
ORIGINAL_FNV = "0xaec368ac6a568632ULL"


def check_layout(ref, trial):
    if len(ref) != len(trial):
        raise ValueError("tensor count differs")
    data = {t.name:t.a for t in trial}
    for r,t in zip(ref,trial):
        if (r.name,r.code)!=(t.name,t.code):
            raise ValueError("tensor names, types or order differ")
        if r.a.shape==t.a.shape:
            continue
        parts=t.name.split(".",2)
        if len(parts)!=3 or parts[0]!="blocks" or not parts[1].isdigit() or int(parts[1])>=12:
            raise ValueError(f"unsupported shape change in {t.name}")
        base=".".join(parts[:2])+".mlp."
        width=data[base+"up.weight.q"].shape[0]
        if width<128 or width>1024 or width%16:
            raise ValueError(f"unsupported MLP width {width}")
        sizes={base+"up.weight.q":(width,192),base+"up.weight.scale":(width,),
               base+"up.bias":(width,),base+"down.weight.q":(192,width)}
        if sizes.get(t.name)!=t.a.shape:
            raise ValueError(f"unsupported shape change in {t.name}")
    for layer in range(12):
        base=f"blocks.{layer}.mlp."
        w=data[base+"up.weight.q"].shape[0]
        if data[base+"up.weight.scale"].size != w or data[base+"down.weight.q"].shape != (192,w):
            raise ValueError(f"inconsistent MLP geometry at layer {layer}")


def stage(candidate, expected, out):
    if out.exists():
        raise ValueError("choose a fresh destination")
    sha, fnv, size=model_id(candidate)
    if sha != expected or hashlib.sha256(BASELINE.read_bytes()).hexdigest()!=REFERENCE_SHA:
        raise ValueError("candidate or reference SHA-256 mismatch")
    check_layout(np2.read_container(str(BASELINE)),np2.read_container(str(candidate)))
    with tempfile.TemporaryDirectory(dir=out.parent) as folder:
        tmp=Path(folder)
        archive=tmp/"sources.tar"
        subprocess.run(["git","-C",str(ROOT),"archive","--output",str(archive),UPSTREAM],check=True)
        source=tmp/"release"
        with tarfile.open(archive) as tar:
            tar.extractall(source,filter="data")
        for name,count in (("src/transformer.h",1),("src/tf1/weights_io_compressed.cpp",1),
                           ("src/tf1/weights_io.h",2)):
            p=source/name
            text=p.read_text()
            if text.count(ORIGINAL_FNV)!=count:
                raise ValueError(f"release fingerprint location changed in {name}")
            p.write_text(text.replace(ORIGINAL_FNV,"0x"+fnv+"ULL"))
        (source/BASELINE.name).write_bytes(candidate.read_bytes())
        source.rename(out)
    print(f"staged {sha} ({size} bytes) at {out}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate",type=Path)
    p.add_argument("expected_sha256")
    p.add_argument("destination",type=Path)
    a=p.parse_args()
    stage(a.candidate.resolve(),a.expected_sha256,a.destination.resolve())
