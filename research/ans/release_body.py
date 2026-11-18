"""Check an ordinary self-extracting release through the first real article bodies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from release_prefix import prefix


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def validate_weight_proof(assembly,proof):
    # Only the published NP2 and the separately established sign-equivalent
    # NP2 have the same continuing-payload oracle used by this experiment.
    originals={"729694cba4969b7d833686ed3f597ff04d3b226da849d5f8004ec020a0ed9563",
               "78ef4ef61a6848e8a3c924eca62c87fe2110586ae8e5323fa303ec4d40d42eaa"}
    if (proof.get("exact_np2_roundtrip") is not True or proof.get("original_sha256") not in originals
        or proof.get("hybrid_sha256")!=assembly.get("weight_sha256")
        or proof.get("hybrid_bytes")!=assembly.get("weight_bytes")
        or assembly.get("reference_archive_sha256")!="88dd7087404122d010261e6855afd88875d34dde1f0539377d3e832fc0f6b3fd"
        or assembly.get("payload_sha256")!="fc2d8c8f824e740ec8b06fa129316043c3876027449005d58df21d0f12254b57"
        or assembly.get("payload_bytes")!=85417870):
        raise ValueError("release lacks a matching proof for the original prediction stream")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("encode","decode"))
    p.add_argument("executable",type=Path)
    p.add_argument("assembly",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--corpus",type=Path)
    p.add_argument("--weights-proof",type=Path)
    p.add_argument("--timeout",type=int,default=12600)
    a=p.parse_args()
    assembly=json.loads(a.assembly.read_bytes())
    here=Path(__file__).resolve().parent
    candidates=[json.loads((here/"hybrid-result.json").read_bytes())["hybrid"],
                json.loads((here/"hybrid-sign-result.json").read_bytes())["assembly"],
                json.loads((here/"residual-result.json").read_bytes())["assembly"],
                json.loads((here/"chained-result.json").read_bytes())["assembly"]]
    if a.weights_proof:
        validate_weight_proof(assembly,json.loads(a.weights_proof.read_bytes()))
    elif assembly not in candidates:
        raise ValueError("provide the matching weight proof for a new release")
    expected_size=assembly["comp_bytes" if a.phase=="encode" else "archive_bytes"]
    if a.executable.stat().st_size!=expected_size:raise ValueError("release file length changed")
    image_sha=digest(a.executable)
    if a.phase=="encode":
        if not a.corpus or digest(a.corpus)!="159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc":
            raise ValueError("ordinary compression requires verified enwik9")
        if image_sha!=assembly["comp_sha256"]:raise ValueError("candidate compressor changed")
        # The complete continuing stream prefix observed at the 20M input checkpoint.
        size,expected=3367414,"52c208ff8ef5e0d0844d928e5a53d0b0411eb456edec3e230ad4d49f4b167a58"
    else:
        if a.corpus or image_sha!=assembly["archive_sha256"]:raise ValueError("candidate archive changed")
        size,expected=20000000,"e3f8084643cb36eac12231108fefc98056d9e415dd7d66fd077d07a6f8796359"
    cpu=min(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpu})
    start=time.monotonic()
    data=prefix(a.executable,a.output,size,a.timeout,a.corpus)
    actual=hashlib.sha256(data).hexdigest()
    if len(data)!=size or actual!=expected:raise ValueError(f"{a.phase} prefix differs: {actual}")
    result=dict(phase=a.phase,executable_sha256=image_sha,one_logical_cpu=cpu,
                canonical_checkpoint_bytes=20000000,matched_bytes=size,matched_sha256=actual,
                elapsed_seconds=time.monotonic()-start)
    (a.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
