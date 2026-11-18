"""Measure the RAM and prediction cost of two- and three-bit semantic keys."""
import argparse
import json
from pathlib import Path
import numpy as np
from numba import njit
from cache_replay import load
from cache_windows import WINDOWS


def mappings(levels):
    levels=np.array(levels,dtype=np.int16)
    raw=np.arange(-127,128,dtype=np.float64)
    # A scalar gain per candidate is free under cosine similarity. Only the
    # chosen 2/3-bit component indices enter the key ring.
    gains=np.array([.72,.9,1.,1.15,1.4])*max(abs(levels))/127
    return np.stack([levels[np.argmin(np.abs(g*raw[:,None]-levels),axis=1)] for g in gains])


@njit
def similarities(q, keys, tables):
    out=np.empty(len(keys))
    qnorm=np.sqrt(np.sum(q*q))
    for i in range(len(keys)):
        best=-1.
        selected=0
        for m in range(len(tables)):
            dot=0;norm=0
            for j in range(keys.shape[1]):
                x=int(keys[i,j]);v=int(tables[m,x+127]);dot+=v*x;norm+=v*v
            value=float(dot)*dot/max(norm,1)
            if value>best:
                best=value;selected=m
        dot=0.;norm=0
        for j in range(keys.shape[1]):
            v=int(tables[selected,int(keys[i,j])+127]);dot+=q[j]*v;norm+=v*v
        out[i]=dot/(qnorm*np.sqrt(norm)) if norm and qnorm else 0.
    return out


def loss(r,chosen,cosine):
    mass=np.exp((cosine-cosine.max())/.020)
    target=mass[r["values"][chosen]==r["target"]].sum()/mass.sum()
    return -np.log2(max(.85*r["tf"]+.15*target,1e-30))


def evaluate(rows,tables):
    delta=[];overlap=[]
    for r in rows:
        sim=similarities(r["q"],r["x"],tables)
        indices=np.argsort(-sim,kind="stable")[:len(r["native"])]
        delta.append(loss(r,indices,sim[indices])-loss(r,r["native"],r["native_scores"]))
        overlap.append(len(set(r["native"])&set(indices))/len(indices))
    return dict(queries=len(rows),delta_bits=float(np.sum(delta)),mean_top32_overlap=float(np.mean(overlap)),
                quarter_deltas_bits=[float(np.sum(s)) for s in np.array_split(delta,4)])


def verify_packing(levels,bits):
    # Four successive lines cover the packed pattern boundaries at 2 and 3 bits.
    codes=(np.arange(128)*7+3)%(1<<bits)
    planes=(codes[:,None]>>np.arange(bits))&1
    payload=np.packbits(planes.ravel(),bitorder="little")
    recovered=np.unpackbits(payload,bitorder="little").reshape(128,bits)@(1<<np.arange(bits))
    if len(payload)!=128*bits//8 or not np.array_equal(codes,recovered) or len(levels)>(1<<bits):
        raise ValueError("fixed-width key round trip failed")
    return len(payload)


def run(data,out,schedule):
    spans=WINDOWS[schedule]
    def rows(start):
        d,info,offset=load(data/f"cache-{start}.bin",data/"results"/f"receipt-{start}.json",data/"output.cmix.temp")
        if offset!=0:raise ValueError("unexpected target coordinate")
        return d,info["trace_sha256"]
    early,_=rows(spans[0][0]);split=int(.75*len(early))
    candidates={2:[[-3,-1,1,3],[-3,-1,0,2],[-2,0,1,3]],
                3:[[-4,-3,-2,-1,0,1,2,3],[-7,-4,-2,-1,0,1,2,4],[-6,-3,-1,0,1,2,3,6],[-7,-4,-2,-1,1,2,4,7]]}
    out.mkdir(exist_ok=True)
    frozen={}
    for bits,options in candidates.items():
        fits=[(evaluate(early[:split],mappings(v))["delta_bits"],v) for v in options]
        best=min(fits)
        frozen[bits]=dict(levels=best[1],candidates_fit_delta_bits=fits,model_selection_queries=split,
                          per_entry_bytes=verify_packing(best[1],bits),original_per_entry_bytes=64,
                          first_span_after_freeze=evaluate(early[split:],mappings(best[1])))
    (out/"frozen.json").write_text(json.dumps(frozen,indent=2)+"\n")
    result=dict(schedule=schedule,physical_ring_entries=2**24,selection=frozen,spans={})
    for start,_ in spans[1:]:
        d,sha=rows(start)
        result["spans"][start]=dict(sha256=sha,measurements={b:evaluate(d,mappings(v["levels"])) for b,v in frozen.items()})
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("data",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--schedule",choices=WINDOWS,default="early");a=p.parse_args();run(a.data,a.output,a.schedule)
