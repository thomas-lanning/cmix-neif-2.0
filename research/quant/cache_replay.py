"""Compare four-bit cache keys on frozen native candidate lists and next-byte labels."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import cache_metric

CANDIDATE=np.dtype([("id","<u4"),("value","u1"),("original","i1",128),("packed","u1",64)])
MIX=np.array([.05,.15,.30])
METHODS=("original","int8","clip","angle","levels","two-blocks","four-blocks",
         "project-quarter","project-half","project-three-quarters","metric-8","metric-32")


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def load(path,receipt,canonical):
    info=json.loads(receipt.read_bytes())
    if digest(path)!=info["trace_sha256"] or path.stat().st_size!=info["trace_bytes"]:
        raise ValueError("cache trace differs from its archive-verified receipt")
    raw=path.read_bytes()
    if raw[:8]!=b"NCACHE01":raise ValueError("unknown cache trace")
    offset=8;rows=[]
    corpus=np.memmap(canonical,dtype="u1",mode="r")
    while offset<len(raw):
        pos,n,k=struct.unpack_from("<3I",raw,offset);offset+=12
        if not info["start"]<=pos<info["stop"] or not 8<=n<=128 or k!=min(n,32):
            raise ValueError("invalid sample metadata")
        def take(dtype,count):
            nonlocal offset
            v=np.frombuffer(raw,dtype=dtype,count=count,offset=offset)
            offset+=v.nbytes
            return v
        q=take("i1",128).astype(np.float64);tf=take("<f4",256).astype(np.float64)
        cand=take(CANDIDATE,n)
        ids=take("<u4",k);cos=take("<f8",k)
        target=raw[offset];offset+=1
        keys=cand["original"].copy();packed=cand["packed"]
        native=np.empty((n,128),dtype=np.int8)
        native[:,::2]=((packed&15).astype(np.int8)^8)-8
        native[:,1::2]=((packed>>4).astype(np.int8)^8)-8
        stock=np.rint(keys.astype(np.float64)*(7/127)).astype(np.int8)
        if not np.array_equal(native,stock):raise ValueError("full keys disagree with the native nibble packing")
        scores=similarity(native,q)
        order=np.argsort(-scores,kind="stable")[:k]
        native_indices=np.array([np.flatnonzero(cand["id"]==i).item() for i in ids])
        if (not np.allclose(scores[order],cos,rtol=0,atol=2e-14) or
                not np.allclose(scores[native_indices],cos,rtol=0,atol=2e-14)):
            raise ValueError("native ranking or cosine not reproduced")
        if not np.isfinite(tf).all() or tf.min()<0 or tf.sum()<=0:raise ValueError("invalid transformer probability")
        rows.append(dict(pos=pos,q=q,x=keys,values=cand["value"].copy(),native=native_indices,
                         target=target,tf=float(tf[target]/tf.sum()),native_scores=cos.copy()))
    if len(rows)!=info["sampled_queries"] or len(rows)<48:
        raise ValueError("insufficient complete sampled queries")
    # Validate the target coordinate independently. Uniqueness guards against
    # silently accepting an off-by-one or the six-byte package header.
    positions=np.array([r["pos"] for r in rows])
    targets=np.array([r["target"] for r in rows])
    shifts=[d for d in range(-8,9) if np.array_equal(corpus[positions+d],targets)]
    if len(shifts)!=1:raise ValueError("target positions differ from canonical bytes")
    return rows,info,shifts[0]


def similarity(keys,query):
    v=keys.astype(np.float64)
    kn=np.sqrt(np.einsum("ij,ij->i",v,v))
    return np.divide((v@query)/max(np.linalg.norm(query),1e-30),kn,out=np.zeros(len(v)),where=kn>0)


def learn(rows):
    # Fixed stride limits repeated keys from large candidate lists.
    data=np.concatenate([r["x"][::13] for r in rows]).astype(np.float64)
    magnitudes=np.bincount((data.astype(np.int16)+127).ravel(),minlength=255)
    x=np.arange(-127,128,dtype=np.float64)
    centers=np.linspace(-120,120,16)
    for _ in range(32):
        group=np.abs(x[:,None]-centers).argmin(axis=1)
        count=np.bincount(group,weights=magnitudes,minlength=16)
        value=np.bincount(group,weights=x*magnitudes,minlength=16)
        centers=np.divide(value,count,out=centers.copy(),where=count>0)
    unit=data/np.maximum(np.linalg.norm(data,axis=1,keepdims=True),1)
    _,axes=np.linalg.eigh(unit.T@unit)
    axes=axes[:,-8:].T.copy()
    for b in axes:
        if b[np.abs(b).argmax()]<0:b*=-1
    integer_axes=np.rint(127*axes/np.max(np.abs(axes),axis=1,keepdims=True)).astype(np.int8)
    return dict(levels=np.rint(centers).astype(int).tolist(),axes=integer_axes.tolist(),
                metrics={f"metric-{n}":cache_metric.fit(rows,n) for n in (8,32)})


def approximation(keys,query,method,model):
    v=keys.astype(np.float64)
    if method=="int8":return v,query
    if method=="original":return np.rint(v*(7/127)),query
    if method.startswith("metric-"):
        weights=np.asarray(model["metrics"][method])
        scale=np.sqrt(np.repeat(weights,query.size//len(weights)))
        return np.rint(v*(7/127))*scale,query*scale
    if method=="clip":return np.clip(np.rint(v*(8/127)),-8,7),query
    if method=="angle":
        candidates=np.clip(np.rint(v[None,:,:]*np.array([7,7.5,8,9,10])[:,None,None]/127),-8,7)
        inner=np.einsum("knj,nj->kn",candidates,v)
        norm=np.einsum("knj,knj->kn",candidates,candidates)
        choose=(inner**2/np.maximum(norm,1)).argmax(axis=0)
        return candidates[choose,np.arange(len(v))],query
    if method=="levels":
        levels=np.asarray(model["levels"])
        return levels[np.searchsorted((levels[:-1]+levels[1:])/2,v)],query
    if method in ("two-blocks","four-blocks"):
        # One nibble per block records its amplitude. The last coordinate
        # supplies that slot, so the key stays at 64 bytes.
        n=2 if method=="two-blocks" else 4
        groups=v.reshape(len(v),n,-1).copy()
        groups[:,:,-1]=0
        peak=np.max(np.abs(groups),axis=2,keepdims=True)
        factor=np.maximum(np.rint(peak*15/127),1)*127/15
        restored=np.clip(np.rint(groups*7/factor),-8,7)*factor
        return restored.reshape(v.shape),query
    strength={"project-quarter":.25,"project-half":.5,"project-three-quarters":.75}[method]
    basis=np.asarray(model["axes"],dtype=np.float64)
    basis/=np.linalg.norm(basis,axis=1,keepdims=True)
    projected=v-strength*((v@basis.T)@basis)
    projected_query=query-strength*((query@basis.T)@basis)
    peak=np.max(np.abs(projected),axis=1,keepdims=True)
    return np.rint(projected*7/np.maximum(peak,1e-20)),projected_query


def evaluate(rows,method,model):
    loss=[];hits=[];overlap=[]
    for r in rows:
        v,query=approximation(r["x"],r["q"],method,model)
        sim=similarity(v,query)
        top=np.argsort(-sim,kind="stable")[:min(32,len(sim))]
        if method=="original":top=r["native"]
        weight=np.exp((sim[top]-np.max(sim[top]))/.020)
        prob=weight[r["values"][top]==r["target"]].sum()/weight.sum()
        loss.append(-np.log2(np.maximum((1-MIX)*r["tf"]+MIX*prob,1e-30)))
        hits.append(r["values"][top[0]]==r["target"])
        overlap.append(len(set(top)&set(r["native"]))/len(top))
    return np.asarray(loss),float(np.mean(hits)),float(np.mean(overlap))


def measure(rows,model):
    base,_,_=evaluate(rows,"original",model)
    metrics={}
    for method in METHODS:
        if method.startswith("metric-") and method not in model.get("metrics",{}):
            continue
        scores,hit,overlap=evaluate(rows,method,model)
        delta=scores-base
        metrics[method]=dict(delta_bits_by_mixture=delta.sum(axis=0).tolist(),
            delta_bits_by_quarter=[float(part.sum()) for part in np.array_split(delta[:,1],4)],
            top_byte_accuracy=hit,original_top32_overlap=overlap)
    return metrics


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("train","test"))
    for name in ("trace","receipt","canonical","model","output"):p.add_argument(name,type=Path)
    a=p.parse_args()
    if a.output.exists() or (a.action=="train" and a.model.exists()):raise SystemExit("choose fresh paths")
    rows,info,shift=load(a.trace,a.receipt,a.canonical)
    if a.action=="train":
        split=3*len(rows)//4
        model=learn(rows[:split]);metrics=measure(rows[split:],model)
        competitors=[s for s in METHODS if s!="int8"]
        model.update(selected=min(competitors,key=lambda s:metrics[s]["delta_bits_by_mixture"][1]),
                     trained_on=info["trace_sha256"],fitted_queries=split,selection_queries=len(rows)-split)
        a.model.write_text(json.dumps(model)+"\n")
    else:
        model=json.loads(a.model.read_bytes())
        if model["trained_on"]==info["trace_sha256"]:raise ValueError("training trace supplied for validation")
        metrics=measure(rows,model)
    report=dict(role=a.action,trace=info,canonical_offset=shift,model_sha256=digest(a.model),
                frozen_selection=model["selected"],cache_mixture_fractions=MIX.tolist(),metrics=metrics)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
