"""Verify upstream's C40 rasters or the full trimmed PPMd training stream in place."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def c40(root):
    published={"dump/paket-zeilen.bin":"0b124db7","dump/zonenmaske.u8":"dc8b1597",
               "dump/match-bytes.uint8":"3b6e8d55","dump/match-len.uint8":"bec3f847",
               "priors/ppmd-probs.float16":"eebb3812","priors/ppmd-bytes.uint8":"2b56a78b",
               "priors/ppmd-quelle.u8":"49729226"}
    hashes={}
    for name,prefix in published.items():
        with (root/name).open("rb") as file: digest=hashlib.file_digest(file,"sha256").hexdigest()
        if not digest.startswith(prefix):raise ValueError(f"{name} differs from upstream's published digest prefix")
        hashes[name]=digest
    return dict(selection="C40",matched_published_hex_digits=8,sha256=hashes)


def full(root):
    cut=13_023_920
    source=root/"priors"
    seen=kept=0
    t_hash=hashlib.sha256(); p_hash=hashlib.sha256()
    with (source/"ppmd-quelle.u8").open("rb") as origins,(source/"ppmd-bytes.uint8").open("rb") as tokens, (source/"ppmd-probs.float16").open("rb") as priors:
        while raw:=origins.read(65536):
            mask=np.frombuffer(raw,dtype="u1")
            if mask.max()>1:raise ValueError("unexpected source tag")
            ids=np.flatnonzero(mask==0)
            t=tokens.read(len(raw));p=priors.read(416*len(raw))
            if len(t)!=len(raw) or len(p)!=416*len(raw):raise ValueError("incomplete raw PPMd raster")
            rows=ids[max(0,cut-seen):]
            seen+=len(ids);kept+=len(rows)
            if len(rows):
                t_hash.update(np.frombuffer(t,dtype="u1")[rows])
                p_hash.update(np.frombuffer(p,dtype="V416")[rows])
        if tokens.read(1) or priors.read(1):raise ValueError("trailing PPMd raster")
    if (seen,kept)!=(501_062_287,488_038_367):raise ValueError("unexpected token counts")
    if t_hash.hexdigest()!="721ad4f3f8a0a7957598cbc62b8ef495e9c620652486123757975b73274568bd":
        raise ValueError("trimmed reference token hash differs")
    if p_hash.hexdigest()!="dc71f3113288cb9aff95adb0cf4f10af353b11888618df7ff187b360f57cbb0a":
        raise ValueError("trimmed reference prior hash differs")
    return dict(selection="full",tokens=kept,prior_bytes=kept*416,token_sha256=t_hash.hexdigest(),
                prior_sha256=p_hash.hexdigest())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("selection",choices=("c40","full"))
    p.add_argument("scratch",type=Path)
    a=p.parse_args()
    print(json.dumps((full if a.selection=="full" else c40)(a.scratch),indent=2))
