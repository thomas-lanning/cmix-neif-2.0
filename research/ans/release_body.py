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


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("encode","decode"))
    p.add_argument("executable",type=Path)
    p.add_argument("assembly",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--corpus",type=Path)
    p.add_argument("--timeout",type=int,default=12600)
    a=p.parse_args()
    assembly=json.loads(a.assembly.read_bytes())
    here=Path(__file__).resolve().parent
    candidates=[json.loads((here/"hybrid-result.json").read_bytes())["hybrid"],
                json.loads((here/"hybrid-sign-result.json").read_bytes())["assembly"]]
    if assembly not in candidates:raise ValueError("release differs from the measured candidate packages")
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
