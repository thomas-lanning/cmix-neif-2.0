"""Fit signed integer weights with an activation-error and entropy penalty."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from allocation import bf16, np2
from evaluate import BASELINE, REFERENCE_SHA, measure


def optimize(original, scales, covariance, penalty):
    w = original.astype(np.float64)
    h = covariance.copy()
    diagonal = h.diagonal().copy()
    h.flat[::len(h)+1] += max(1e-9, diagonal.mean()*.001)
    diag = h.diagonal()
    if np.abs(w).max()>7 or min(diag)<=0:
        raise ValueError("expected signed four-bit weights and a regularized covariance")
    values = np.arange(-7,8)
    counts = np.bincount(original.ravel().astype(np.int16)+7,minlength=15) + 1
    price = -np.log2(counts/counts.sum())
    # The price is one model of the storage cost. The native container size
    # decides what the exporter actually saved.
    physical = np.square(bf16(scales).astype(np.float64))
    positive = physical[physical>0]
    if not len(positive):
        return original.copy()
    weight = penalty * diag.mean()*np.median(positive) / np.maximum(physical,1e-30)
    prices = weight[:,None]*price
    peak = np.abs(w).max(axis=1)
    prices[np.abs(values)[None,:] > peak[:,None]] = np.inf
    q = original.copy()
    residual = np.zeros_like(w)  # (w-q) H
    for order in (range(w.shape[1]),reversed(range(w.shape[1]))):
        for j in order:
            target = q[:,j] + residual[:,j]/diag[j]
            costs = diag[j]*np.square(target[:,None]-values) + prices
            proposed = values[np.argmin(costs,axis=1)]
            changed = np.flatnonzero(proposed!=q[:,j])
            if len(changed):
                step = proposed[changed]-q[changed,j]
                residual[changed] -= step[:,None]*h[j]
                q[changed,j] = proposed[changed]
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("replay","suite","gram","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--until",type=float,default=float("inf"))
    p.add_argument("--penalties",nargs="+",type=float,default=[.04,.16,.64,.01,.32,1.28])
    a=p.parse_args()
    a.replay,a.suite,a.gram,a.output=(x.resolve() for x in (a.replay,a.suite,a.gram,a.output))
    if min(a.penalties)<=0:
        raise ValueError("penalties must be positive")
    manifest=(a.gram/"manifest.json").read_bytes()
    if json.loads(manifest)["model_sha256"]!=REFERENCE_SHA or hashlib.sha256(BASELINE.read_bytes()).hexdigest()!=REFERENCE_SHA:
        raise ValueError("published model hash differs")
    screen=a.output/"screen"
    screen.mkdir(exist_ok=True,parents=True)
    provenance=hashlib.sha256(manifest+Path(__file__).read_bytes()+(a.suite/"manifest.json").read_bytes()
                              +a.replay.read_bytes()+Path(__file__).with_name("evaluate.py").read_bytes()).hexdigest()
    base=np2.read_container(str(BASELINE))
    names={t.name:t.a for t in base}
    for penalty in a.penalties:
        variants={}
        for family in ("down","up"):
            for layer in range(16):
                if time.time()>=a.until: return
                name=f"blocks.{layer}.mlp.{family}.weight."
                cov=np.load(a.gram/f"layer-{layer:02d}-{family}.npy")
                variants[name+"q"]=optimize(names[name+"q"],names[name+"scale"].reshape(-1),cov,penalty)
        for families,layers in ((["up","down"],range(16)),(["down"],range(16)),(["up"],range(16)),
                                (["up","down"],range(4)),(["up","down"],range(4,8)),
                                (["up","down"],range(8,12)),(["up","down"],range(12,16))):
            if time.time()>=a.until: return
            spec=dict(kind="entropy-penalty",families=families,layers=list(layers),penalty=penalty)
            key=hashlib.sha256(json.dumps([provenance,spec],sort_keys=True).encode()).hexdigest()[:20]
            dest=screen/(key+".json")
            if dest.exists(): continue
            start=time.monotonic()
            edits={f"blocks.{layer}.mlp.{family}.weight.q" for family in families for layer in layers}
            changed=sum(np.count_nonzero(variants[n]!=names[n]) for n in edits)
            if not changed: continue
            tensors=[np2.Ent(t.name,t.code,variants[t.name] if t.name in edits else t.a) for t in base]
            encoded,_,expected=np2.encode_np(tensors,True,np2.np_groups(tensors,True,True),True)
            with tempfile.TemporaryDirectory(dir=screen) as td:
                tmp=Path(td); model=tmp/"model.c5"; model.write_bytes(encoded)
                recovered=np2.read_container(str(model))
                if len(expected)!=len(recovered) or not all(t.same(u) for t,u in zip(expected,recovered)):
                    raise ValueError("model changed during NP2 export")
                r=measure(a.replay,a.suite,model,a.output/"discovery",key)
                record={k:r[k] for k in ("candidate_sha256","candidate_bytes","twice_weight_delta","search_proxy_bytes",
                                         "confirmation_proxy_bytes","proxy_delta_bytes","bits_per_target_delta_by_window","replay_seconds")}
                record.update(key=key,specs=[spec],changed=int(changed),provenance=provenance,total_seconds=time.monotonic()-start)
                model.replace(screen/(key+".c5"))
                file=tmp/"score.json"; file.write_text(json.dumps(record,indent=2)+"\n"); file.replace(dest)
                print(key,changed,r["twice_weight_delta"],r["search_proxy_bytes"],r["confirmation_proxy_bytes"],flush=True)


if __name__=="__main__":
    main()
