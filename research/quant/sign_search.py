"""Greedily minimize actual NP2 bytes using equivalent signed-row representations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from allocation import bf16, np2
from evaluate import BASELINE, REFERENCE_SHA


def same_model(reference, other):
    """Check values, order and every tensor outside the signed row pairs."""
    lookup={t.name:t for t in other}
    original={t.name:t for t in reference}
    if list(original)!=list(lookup):return False
    checked=set()
    for t in reference:
        if t.name in checked:continue
        u=lookup[t.name]
        if t.same(u):continue
        if not t.name.endswith((".weight.q",".weight.scale")):return False
        base=t.name[:t.name.rfind(".")+1]
        q,s=original[base+"q"],original[base+"scale"]
        a,b=lookup[q.name],lookup[s.name]
        if q.a.shape!=a.a.shape or s.a.shape!=b.a.shape or q.code!=a.code or s.code!=b.code:return False
        if not np.array_equal(q.m2()*bf16(s.a).reshape(-1,1),a.m2()*bf16(b.a).reshape(-1,1)):return False
        checked.update((q.name,s.name))
    return True


def row_counts(matrix):
    """Exact NP2 width bucket of each symbol, skipping its all-zero rows."""
    wc=np2.wc
    raw=matrix.astype(np.int64)
    used=np.any(raw,axis=1)
    q=raw[used]
    if not len(q):return np.zeros((len(raw),wc.W_NB,15),dtype=np.int32)
    absolute=np.abs(q)
    hist=absolute.cumsum(axis=1)-absolute
    cols=absolute.cumsum(axis=0)-absolute
    prior=absolute.ravel().cumsum().reshape(q.shape)-absolute
    n,j=np.indices(q.shape)
    m=np.maximum((prior*65536+64*wc.W_M0)//(n*q.shape[1]+j+64),1)
    r=(hist*65536+wc.W_AROW*m)//(j+wc.W_AROW)
    c=(cols*65536+wc.W_ACOL*m)//(n+wc.W_ACOL)
    bucket=np.clip((8*r*c)//(m*65536)-8,0,wc.W_NB-1)
    idx=(n*wc.W_NB+bucket)*15+q+7
    counts=np.zeros((len(raw),wc.W_NB,15),dtype=np.int32)
    counts[used]=np.bincount(idx.ravel(),minlength=len(q)*wc.W_NB*15).reshape(len(q),wc.W_NB,15)
    return counts


def optimize(counts, negative, method):
    # Conditional symbol probabilities guide proposals. Only a smaller actual
    # encoded file with identical decoded effective tensors can be retained.
    values=np.arange(-7,8)
    orientation=(counts.sum(axis=1)@values)<0
    if method=="sparse": orientation &= np.abs(counts.sum(axis=1)@values)>counts.sum(axis=(1,2))*.12
    if method=="reverse":orientation=~orientation
    contrast=counts[:,:,::-1]-counts
    for _ in range(15):
        sums=(counts+contrast*orientation[:,None,None]).sum(axis=0)+2
        prices=-np.log2(sums/sums.sum(axis=1,keepdims=True))
        delta=np.einsum("rbs,bs->r",contrast,prices)
        signs=negative^orientation
        logodds=np.log2((signs.sum()+1)/((~signs).sum()+1))
        choose=delta<np.where(negative,-logodds,logodds)
        if np.array_equal(choose,orientation):break
        orientation=choose
    benefit=float(np.maximum(0,np.where(negative,-logodds,logodds)-delta)[orientation].sum())
    return orientation,benefit


def atomic_json(path,obj):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2)+"\n");tmp.replace(path)


def run(a):
    if hashlib.sha256(BASELINE.read_bytes()).hexdigest()!=REFERENCE_SHA:raise ValueError("wrong baseline")
    original=np2.read_container(str(BASELINE))
    a.output.mkdir(parents=True,exist_ok=True)
    best=a.output/"best.c5"
    seed=a.seed or BASELINE
    if not best.exists():best.write_bytes(seed.read_bytes())
    bank=np2.read_container(str(best))
    if not same_model(original,bank):raise ValueError("seed differs in effective values or order")
    index={t.name:i for i,t in enumerate(bank)}
    groups=np2.np_groups(bank,True,True)
    keys={g.key for g in groups}
    ledger=a.output/"trials.jsonl"
    tried={r["trial"] for r in map(json.loads,ledger.read_text().splitlines())} if ledger.exists() else set()
    for method in ("dense","sparse","reverse"):
        proposals=[]
        for i,t in enumerate(bank):
            if i in keys or not t.name.endswith(".weight.q") or t.code!=0: continue
            pair=index.get(t.name[:-1]+"scale")
            if pair is None or bank[pair].code!=1 or bank[pair].a.size!=t.rows():continue
            mask,benefit=optimize(row_counts(t.m2()),(bank[pair].a.reshape(-1)&32768)!=0,method)
            if mask.any():proposals.append((benefit,i,pair,mask))
        for _,i,pair,mask in sorted(proposals,key=lambda x:-x[0]):
            if time.time()>=a.until or len(tried)>=a.trials:return
            trial=f"{method}/{bank[i].name}"
            if trial in tried:continue
            q=bank[i].m2(); s=bank[pair].a.reshape(-1)
            q[mask]*=-1;s[mask]^=np.uint16(32768)
            blob,_,permuted=np2.encode_np(bank,True,groups,True)
            ordered=all(t.same(v) for t,v in zip(bank,permuted,strict=True))
            old=best.stat().st_size;keep=ordered and len(blob)<old
            model_hash=hashlib.sha256(blob).hexdigest()
            if keep:
                tmp=a.output/"candidate.tmp";tmp.write_bytes(blob)
                recovered=np2.read_container(str(tmp))
                if not same_model(original,recovered) or not all(t.same(v) for t,v in zip(bank,recovered,strict=True)):
                    raise ValueError("candidate values or execution order differ")
                tmp.replace(best)
                atomic_json(a.output/"best.json",dict(bytes=len(blob),sha256=model_hash,baseline_bytes=BASELINE.stat().st_size,
                    seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),effective_tensors_identical=True))
            else:
                q[mask]*=-1;s[mask]^=np.uint16(32768)
            r=dict(trial=trial,candidate_bytes=len(blob),candidate_sha256=model_hash,previous_bytes=old,keep=keep,
                   same_neuron_order=ordered,changed_rows=int(mask.sum()),current_bytes=best.stat().st_size)
            with ledger.open("a") as f:print(json.dumps(r),file=f)
            tried.add(trial)
            print(trial,len(blob)-old,keep,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path);p.add_argument("--seed",type=Path)
    p.add_argument("--trials",type=int,default=80);p.add_argument("--until",type=float,default=float("inf"))
    run(p.parse_args())
