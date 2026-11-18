"""Freeze a quantized logistic correction on one trace and evaluate later traces."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

EVENT=np.dtype([("final","<u4"),("floor","<u4"),("head","<u4"),("base","<u4"),
                ("prefix","<u2"),("previous","u1"),("bit","u1")])
TOTAL=2**32
SCALE=1024


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def features(directory, receipt):
    info=json.loads(receipt.read_bytes())
    path=directory/f"span-{info['start']}.bin"
    if digest(path)!=info["trace_sha256"] or path.stat().st_size!=info["events"]*EVENT.itemsize:
        raise ValueError("trace differs from the verified span")
    ev=np.memmap(path,mode="r",dtype=EVENT)
    if np.max(ev["bit"])>1:raise ValueError("invalid bit")
    def logit(key):
        p=ev[key].astype(np.float64)/TOTAL
        if not np.all((p>0)&(p<1)):raise ValueError("invalid model probability")
        return np.log(p)-np.log1p(-p)
    base=logit("final");floor=logit("floor");head=logit("head")
    x=np.stack([np.ones(len(ev)),np.clip(base,-15,15),np.clip(floor-base,-6,6),
                np.clip(head-floor,-6,6),np.clip(logit("base")-head,-6,6)],axis=1)
    depth=np.minimum(7,np.floor(np.log2(np.maximum(1,ev["prefix"]))).astype(np.int16))
    confidence=np.searchsorted([1.,3.,6.],np.abs(base))
    return dict(event=ev,base=base,x=x,depth=depth,confidence=confidence)


def grouping(data, scope):
    if scope=="global":return np.zeros_like(data["depth"]),1
    if scope=="depth":return data["depth"],8
    return 4*data["depth"]+data["confidence"],32


def fit(data, which, scope, penalty, stages):
    groups,n=grouping(data,scope)
    dim=5 if stages else 2
    fitted=np.zeros((n,dim),dtype=np.int16)
    for g in range(n):
        ix=which[(groups[which]==g)&(data["event"]["prefix"][which]!=0)]
        if len(ix)<dim*4:continue
        x=data["x"][ix,:dim];base=data["base"][ix];y=data["event"]["bit"][ix]
        beta=np.zeros(dim)
        for _ in range(5):
            p=1/(1+np.exp(-np.clip(base+x@beta,-40,40)))
            hess=(x.T*(p*(1-p)))@x
            hess.flat[::dim+1]+=penalty
            update=np.linalg.solve(hess,x.T@(y-p)-penalty*beta)
            beta=np.clip(beta+update,-1.,1.)
            if np.max(np.abs(update))<1e-5:break
        fitted[g]=np.rint(beta*SCALE).astype(np.int16)
    return dict(scope=scope,penalty=penalty,stages=stages,scale=SCALE,
                coefficient_bytes=fitted.nbytes,coefficients=fitted.tolist())


def predicted(data, model, ix):
    coeff=np.asarray(model["coefficients"])
    grp,_=grouping(data,model["scope"])
    delta=np.einsum("ij,ij->i",data["x"][ix,:coeff.shape[1]],coeff[grp[ix]])/model["scale"]
    logits=data["base"][ix]+delta
    p=np.clip(np.rint(TOTAL/(1+np.exp(-np.clip(logits,-45,45)))),1,TOTAL-1).astype(np.uint32)
    bypass=data["event"]["prefix"][ix]==0
    p[bypass]=data["event"]["final"][ix[bypass]]
    return p


def bits(p, bit):
    p=p.astype(np.float64)/TOTAL
    return -np.sum(np.where(bit,np.log2(p),np.log2(1-p)),dtype=np.float64)


def arithmetic_size(p, bit):
    lo=0;hi=TOTAL-1;size=1
    for probability,symbol in zip(p.tolist(),bit.tolist()):
        mid=lo+((hi-lo)*probability>>32)
        if symbol:hi=mid
        else:lo=mid+1
        while lo>>24==hi>>24:
            size+=1;lo=(lo<<8)&0xffffffff;hi=((hi<<8)&0xffffffff)|255
    return size


def freeze(data,path,trace_hash):
    n=len(data["base"]);first=np.arange(n//2);second=np.arange(n//2,n)
    target=data["event"]["bit"][second]
    unchanged=bits(data["event"]["final"][second],target)
    ranking=[]
    for scope in ("global","depth","depth-confidence"):
        for penalty in (32,256,2048):
            for stages in (False,True):
                m=fit(data,first,scope,penalty,stages)
                loss=bits(predicted(data,m,second),target)
                ranking.append(dict(scope=scope,penalty=penalty,stages=stages,confirmation_delta_bits=loss-unchanged,
                                    coefficient_bytes=m["coefficient_bytes"]))
    # Choose on the second, chronological half of the training trace; later
    # traces never participate. The simpler fit wins an exact tie.
    ranking.sort(key=lambda r:(r["confirmation_delta_bits"],r["coefficient_bytes"]))
    winner=ranking[0]
    model=fit(data,np.arange(n),winner["scope"],winner["penalty"],winner["stages"])
    record=dict(training_trace=trace_hash,training_events=n,choices=ranking,model=model)
    path.write_text(json.dumps(record,indent=2)+"\n")
    return record


def run(a):
    a.output.mkdir(parents=True,exist_ok=True)
    frozen=a.output/"frozen.json"
    for start in (16000000,20000000,26000000,34000000,42000000):
        receipt=a.verified/f"{start}.json"
        while not receipt.exists():
            if time.time()>a.until:return
            time.sleep(30)
        trial=a.output/f"{start}.json"
        if trial.exists():continue
        data=features(a.traces,receipt)
        ref=json.loads(receipt.read_bytes())
        if not frozen.exists():
            if start!=16000000:raise ValueError("missing training trace")
            freeze(data,frozen,ref["trace_sha256"])
        study=json.loads(frozen.read_bytes())
        if study["training_trace"]!=json.loads((a.verified/"16000000.json").read_bytes())["trace_sha256"]:
            raise ValueError("training reference changed")
        span=np.arange(len(data["base"]))
        pred=predicted(data,study["model"],span)
        bit=data["event"]["bit"]
        base=data["event"]["final"]
        origin=arithmetic_size(base,bit);actual=arithmetic_size(pred,bit)
        result=dict(**ref,role="training" if start==16000000 else "held-out",
                    frozen_sha256=digest(frozen),original_arithmetic_bytes=origin,adjusted_arithmetic_bytes=actual,
                    delta_bytes=actual-origin,model_table_bytes=study["model"]["coefficient_bytes"],
                    delta_ideal_quarter_bytes=[(bits(pred[ix],bit[ix])-bits(base[ix],bit[ix]))/8 for ix in np.array_split(span,4)])
        temp=trial.with_suffix(".tmp");temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(trial)
        print(start,result["role"],origin,actual,actual-origin,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("traces","verified","output"):p.add_argument(n,type=Path)
    p.add_argument("--until",type=float,default=float("inf"))
    run(p.parse_args())
