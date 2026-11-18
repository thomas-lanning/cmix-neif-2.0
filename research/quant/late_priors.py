"""Verify the complete training prior stream and save late article slices in one native pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from compact_priors import ROW_BYTES


def coalesce(spans):
    result = []
    for lo,hi in sorted(spans):
        if hi<=lo or lo<0:
            raise ValueError("invalid interval")
        if result and lo<=result[-1][1]:
            result[-1][1]=max(hi,result[-1][1])
        else:
            result.append([lo,hi])
    return result


class Collector:
    def __init__(self, plans, output, cut):
        self.output=output
        self.cut=cut
        self.request=coalesce(plans)
        self.ends=np.array([r[1] for r in self.request])
        self.raw=0
        self.stream=0
        self.kept=0
        self.selected=[]
        self.tokens_hash=hashlib.sha256()
        self.trimmed_hash=hashlib.sha256()
        self.raw_hash=hashlib.sha256()

    def add(self, probabilities, tokens, origins):
        if len(probabilities)!=len(origins)*ROW_BYTES or len(tokens)!=len(origins):
            raise ValueError("different lengths across native streams")
        self.raw_hash.update(probabilities)
        mask=np.frombuffer(origins,dtype="u1")
        if np.any(mask>1):
            raise ValueError("invalid origin")
        actual=np.flatnonzero(mask==0)
        n=len(actual)
        values=np.frombuffer(probabilities,dtype=f"V{ROW_BYTES}")
        keep=actual[max(0,self.cut-self.stream):]
        if len(keep):
            self.tokens_hash.update(np.frombuffer(tokens,dtype="u1")[keep])
            self.trimmed_hash.update(values[keep])
            self.kept+=len(keep)
        index=self.ends.searchsorted(self.stream,side="right")
        for low,high in self.request[index:] if n else ():
            if low>=self.stream+n:
                break
            lo=int(actual[max(0,low-self.stream)])
            hi=int(actual[min(n,high-self.stream)-1])+1
            self.output.write(memoryview(probabilities)[lo*ROW_BYTES:hi*ROW_BYTES])
            if self.selected and self.selected[-1][1]==self.raw+lo:
                self.selected[-1][1]=self.raw+hi
            else:
                self.selected.append([self.raw+lo,self.raw+hi])
        self.raw+=len(origins)
        self.stream+=n

    def receipt(self):
        return dict(prior_rows=self.raw,stream_tokens=self.stream,trimmed_tokens=self.kept,
                    token_sha256=self.tokens_hash.hexdigest(),trimmed_prior_sha256=self.trimmed_hash.hexdigest(),
                    raw_sha256=self.raw_hash.hexdigest(),spans=self.selected)


def collect(folder,plans,destination,command):
    prob=destination/"selected.float16"
    t=folder/"ppmd-bytes.uint8"
    o=folder/"ppmd-quelle.u8"
    if any(p.exists() for p in (prob,t,o)):
        raise ValueError("use a fresh destination")
    read_fd,write_fd=os.pipe()
    env=dict(os.environ,NEIF_TF1_SAVE_PRIORS=f"/proc/self/fd/{write_fd}",
             NEIF_PPMD_ONLY_BYTES=str(t),NEIF_PPMD_ONLY_QUELLE=str(o))
    with os.fdopen(read_fd,"rb") as source,os.fdopen(write_fd,"wb") as sink,prob.open("xb") as out:
        c=Collector(plans,out,13_023_920)
        with subprocess.Popen(command,env=env,pass_fds=(write_fd,)) as child:
            sink.close()
            try:
                pending=bytearray()
                # Native token/origin writers flush in blocks of 65,536. Keep
                # consuming the prior pipe while those smaller files catch up.
                while True:
                    data=source.read(ROW_BYTES*16384)
                    if data:
                        pending.extend(data)
                    else:
                        if child.wait():
                            raise ValueError("native prior generation failed")
                    if t.is_file() and o.is_file():
                        available=min(t.stat().st_size,o.stat().st_size)-c.raw
                        n=min(len(pending)//ROW_BYTES,available)
                        if n>0:
                            with t.open("rb") as tf,o.open("rb") as of:
                                tf.seek(c.raw);of.seek(c.raw)
                                c.add(pending[:n*ROW_BYTES],tf.read(n),of.read(n))
                            del pending[:n*ROW_BYTES]
                    if not data:
                        break
                    if len(pending)>192<<20:
                        raise ValueError("native origin files fell too far behind the prior stream")
                if pending or t.stat().st_size!=c.raw or o.stat().st_size!=c.raw:
                    raise ValueError("truncated native streams")
            finally:
                if child.poll() is None:
                    child.kill()
    r=c.receipt()
    if (r["stream_tokens"],r["trimmed_tokens"],r["token_sha256"],r["trimmed_prior_sha256"])!=(
            501062287,488038367,"721ad4f3f8a0a7957598cbc62b8ef495e9c620652486123757975b73274568bd",
            "dc71f3113288cb9aff95adb0cf4f10af353b11888618df7ff187b360f57cbb0a"):
        raise ValueError(f"full training digest differs: {r['stream_tokens']}, {r['trimmed_prior_sha256']}")
    if prob.stat().st_size!=sum(hi-lo for lo,hi in c.selected)*ROW_BYTES:
        raise ValueError("incomplete saved slices")
    (destination/"raw-plan.json").write_text(json.dumps(r,indent=2)+"\n")
    print("verified complete corpus",r["trimmed_tokens"],r["trimmed_prior_sha256"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--additional-plan", action="append", default=[], type=Path)
    p.add_argument("folder",type=Path)
    p.add_argument("plan",type=Path)
    p.add_argument("second_plan",type=Path)
    p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args()
    plans=[span for path in (a.plan, a.second_plan, *a.additional_plan)
           for span in json.loads(path.read_bytes())["source_spans"]]
    collect(a.folder.resolve(),plans,a.folder.resolve(),a.command)
