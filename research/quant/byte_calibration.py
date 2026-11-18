"""Fit small integer probability tables on native byte scores from disjoint articles."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import REFERENCE_SHA, ROOT
from fixtures import MAGIC, ROW
from head_fit import digest


def byte_classes():
    bitmap=(ROOT/"training/tf/2_package/vocab-bitmap.u8").read_bytes()
    raw=[v for v in range(256) if bitmap[v//8] & (1<<(v%8))]
    if len(raw)!=208:
        raise ValueError("unexpected native vocabulary")
    def group(v):
        if 97<=v<=122: return 0 if v in b"aeiou" else 1
        if 65<=v<=90: return 2
        if 48<=v<=57: return 3
        if v==32: return 4
        if v in b"\r\n\t": return 5
        if v in b"[]{}|<>=/": return 6
        if v<128: return 7
        return 8+(v-128)//32
    return np.array([group(v) for v in raw])


def documents(suite,profile):
    m=json.loads((suite/"manifest.json").read_bytes())
    p=json.loads((profile/"manifest.json").read_bytes())
    if p["model"]!=REFERENCE_SHA or p["suite"]!=digest(suite/"manifest.json"):
        raise ValueError("wrong native capture")
    classes=byte_classes()
    out=[]
    for w,f in zip(m["windows"],p["files"],strict=True):
        fp=profile/f["file"];wp=suite/w["file"]
        if digest(wp)!=w["sha256"] or digest(fp)!=f["sha256"] or wp.stem!=fp.stem:
            raise ValueError("native inputs changed")
        raw=wp.read_bytes()
        if not raw.startswith(MAGIC):
            raise ValueError("invalid fixture")
        rows=np.frombuffer(raw,dtype=ROW,offset=len(MAGIC))
        rows=rows[rows["score"]!=0]
        prob=np.memmap(fp,dtype="<f4",mode="r").reshape(-1,400)[:,192:].astype(np.float64)
        if len(prob)!=len(rows) or len(prob)!=w["targets"] or np.max(np.abs(prob.sum(1)-1))>1e-5:
            raise ValueError("profile and targets differ")
        prob/=prob.sum(1,keepdims=True)
        prior=rows["prior"].view("<f2").astype(np.float32)
        # Both sides are available before the next symbol is read.
        confidence=prob>=1/16
        agree=(prior>=prob/4).astype(int)+(prior>=4*prob).astype(int)
        ids=np.arange(len(rows)); target=rows["target"]
        cells=(classes[None,:]*2+confidence)*3+agree
        nc=len(np.unique(classes))*6
        mass=np.bincount((ids[:,None]*nc+cells).ravel(),weights=prob.ravel(),minlength=len(rows)*nc).reshape(-1,nc)
        out.append((w["document"],mass.astype("f4"),cells[ids,target],prob[ids,target]))
    return out


def grouped(docs,scope):
    base=np.concatenate([d[1] for d in docs]).reshape(-1,12,2,3)
    label=np.concatenate([d[2] for d in docs])
    if scope=="class":
        return base.sum((2,3)),label//6
    if scope=="class-and-confidence":
        return base.sum(3).reshape(len(base),-1),label//3
    if scope=="class-and-prior":
        return base.sum(2).reshape(len(base),-1),label//6*3+label%3
    return base.reshape(len(base),-1),label


def fit(docs,scope,penalty):
    mass,label=grouped(docs,scope)
    mass=mass.astype(np.float64)
    beta=np.zeros(mass.shape[1])
    counts=np.bincount(label,minlength=len(beta))
    for _ in range(10):
        pred=mass*np.exp(beta)
        pred/=pred.sum(1,keepdims=True)
        sums=pred.sum(0)
        update=np.linalg.solve(np.diag(sums+penalty)-pred.T@pred,counts-sums-penalty*beta)
        beta+=np.clip(update,-.25,.25)
        if np.max(np.abs(update))<1e-6: break
    return np.clip(np.rint(128*np.expm1(beta)),-48,96).astype("i1")


def score(docs,scope,table):
    f=1+np.asarray(table,dtype=float)/128
    rows=[]
    for doc in docs:
        mass,label=grouped([doc],scope)
        prob=doc[3]
        changed=prob*f[label]/(mass@f)
        delta=np.log2(np.maximum(prob,3e-8)/np.maximum(changed,3e-8))
        rows.append(dict(document=doc[0],targets=len(delta),delta_bits=float(delta.sum())))
    return rows


def run(a):
    fit_docs=documents(a.suite,a.profile)
    if a.screen is None:
        train,select=fit_docs[::2][:4],fit_docs[::2][4:]
        screen=fit_docs[1::2]
    else:
        screen=documents(a.screen,a.screen_profile)
        if {r[0] for r in screen}&{r[0] for r in fit_docs}:
            raise ValueError("training and screening overlap")
        train=[d for i,d in enumerate(fit_docs) if i%6!=5]
        select=fit_docs[5::6]
    candidates=[]
    for scope in ("class","class-and-confidence","class-and-prior","all"):
        for penalty in (32,256,2048):
            table=fit(train,scope,penalty)
            metric=sum(x["delta_bits"] for x in score(select,scope,table))
            candidates.append((metric,len(table),scope,penalty,table))
    best=min(candidates,key=lambda x:x[:2])
    scope=best[2];table=best[4]
    if best[0]>=0:table=np.zeros(len(table),"i1")
    measured=score(screen,scope,table)
    r=dict(fit_documents=[r[0] for r in train],selection_documents=[r[0] for r in select],
           candidates=[c[:4] for c in candidates],scope=scope,table=table.tolist(),selection_bits=best[0],
           source=digest(a.suite/"manifest.json"),screen=measured,screen_bits=sum(r["delta_bits"] for r in measured))
    if a.validation:
        validation=documents(a.validation,a.validation_profile)
        if {d[0] for d in validation}&{d[0] for d in train+select+screen}:
            raise ValueError("validation overlaps fitted articles")
        r["independent"]=score(validation,scope,table)
        r["independent_bits"]=sum(v["delta_bits"] for v in r["independent"])
    a.output.write_text(json.dumps(r,indent=2)+"\n")
    print(scope,len(table),r["selection_bits"],r["screen_bits"],r.get("independent_bits"),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("suite","profile","output"):p.add_argument(name,type=Path)
    for name in ("screen","screen-profile","validation","validation-profile"):p.add_argument("--"+name,type=Path)
    run(p.parse_args())
