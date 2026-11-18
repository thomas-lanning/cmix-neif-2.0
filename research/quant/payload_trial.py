"""Compare a frozen quantization export with the real mixed-coder prefix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from evaluate import BASELINE, model_id
from validate_exports import VALIDATION


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def choose(exports, validation, sha):
    record=json.loads((validation/"summary.json").read_bytes())
    if record["validation_suite"]!=VALIDATION:
        raise ValueError("unknown independent validation fixture")
    row=next((r for r in record["outcomes"] if r["candidate_sha256"]==sha),None)
    if not row or row["projected_entry_delta_bytes"]>=0:
        raise ValueError("candidate did not improve the independent article estimate")
    paths=[p for p in exports.rglob("*.c5") if digest(p)==sha]
    if not paths:
        raise ValueError("the frozen model was absent from the original quick-screen artifact")
    return paths[0],row


def encode(exe, corpus, model, folder, quick):
    folder.mkdir()
    shutil.copyfile(BASELINE.parent/"dictionary/english.dic",folder/".dict")
    shutil.copyfile(model,folder/BASELINE.name)
    sha,fnv,size=model_id(model)
    env=dict(os.environ,NEIF_CANDIDATE_FNV=fnv,NEIF_CANONICAL_STREAM=str(corpus))
    if quick:
        env["NEIF_QUICK_CHECK"]="1"
    start=time.monotonic()
    with (folder/"codec.log").open("wb") as log:
        subprocess.run([exe,"-e",corpus,"payload"],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return dict(model_sha256=sha,weight_bytes=size,elapsed_seconds=time.monotonic()-start,
                encoded_bytes=(folder/"payload").stat().st_size,encoded_sha256=digest(folder/"payload"))


def run(a):
    import struct
    if digest(a.canonical)!="c35d3073eeba70f22a2e08a50202b53eb95466e3e9f1f2702f6f9b3f96fd8718":
        raise ValueError("unexpected canonical corpus")
    if digest(a.archive)!="88dd7087404122d010261e6855afd88875d34dde1f0539377d3e832fc0f6b3fd":
        raise ValueError("unexpected reference archive")
    candidate,qualifier=choose(a.exports,a.validation,a.sha)
    a.out.mkdir()
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    control=encode(a.exe,a.canonical,BASELINE,a.out/"control",True)
    actual=(a.out/"control/payload").read_bytes()[:-1]
    with a.archive.open("rb") as f:
        f.seek(-8,2);length,weights=struct.unpack("<II",f.read(8))
        f.seek(-length-8,2);expected=f.read(len(actual))
    if weights!=BASELINE.stat().st_size or len(actual)<=4096 or actual!=expected:
        raise ValueError("the native evaluator's first 200k input bytes failed the published bitstream comparison")
    result=encode(a.exe,a.canonical,candidate,a.out/"candidate",False)
    base=json.loads(Path(__file__).with_name("prefix-calibration.json").read_bytes())
    def parse(lines):
        return {int(pos):int(n) for pos,n in (line.split() for line in lines)}
    reference=parse(base["checkpoints"])
    observed=parse((a.out/"candidate/prefix-sizes.tsv").read_text().splitlines())
    if set(reference)!=set(observed):
        raise ValueError("incomplete candidate checkpoints")
    delta={k:observed[k]-v for k,v in reference.items()}
    if a.sha=="78ef4ef61a6848e8a3c924eca62c87fe2110586ae8e5323fa303ec4d40d42eaa":
        # This specific model has a tensor-by-tensor equivalence proof.
        if result["encoded_sha256"]!=base["reference_payload_sha256"]:
            raise ValueError("the equivalent-row positive control did not reproduce the native reference bytes")
    worker=dict(cpu=cpu,verified_control_bytes=len(actual),control=control,candidate=result,
                independent_articles=qualifier,real_payload_delta_bytes=delta,
                delta_article_portion=delta[20000000]-delta[15282108],
                both_weight_copies_delta=2*(result["weight_bytes"]-control["weight_bytes"]),
                end_checkpoint_canonical_bytes=20000000)
    (a.out/"measurement.json").write_text(json.dumps(worker,indent=2)+"\n")
    print(json.dumps(worker),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("exe","canonical","archive","exports","validation","out"):
        p.add_argument(name,type=lambda s:Path(s).resolve())
    p.add_argument("sha")
    run(p.parse_args())
