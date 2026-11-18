"""Recover a frozen quantized model by fitting its output head to real token targets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from allocation import bf16, np2, round_bf16
from distill import Adam
from evaluate import BASELINE, REFERENCE_SHA, measure, model_id
from fixtures import MAGIC, ROW


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def collect(exe,suite,model,out):
    manifest=suite/"manifest.json"
    source=json.loads(manifest.read_bytes())
    sha,fnv,_=model_id(model)
    out.mkdir(exist_ok=False,parents=True)
    files=[]
    for win in source["windows"]:
        p=suite/win["file"]
        if digest(p)!=win["sha256"]: raise ValueError("fixture hash differs")
        destination=out/(p.stem+".f32")
        got=json.loads(subprocess.check_output([exe,model,fnv,p,destination],text=True,timeout=300))
        if got["targets"]!=win["targets"] or destination.stat().st_size!=got["targets"]*400*4:
            raise ValueError("incorrect head profile length")
        files.append(dict(file=destination.name,sha256=digest(destination)))
    meta=dict(model=sha,suite=digest(manifest),executable=digest(exe),files=files)
    (out/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")


def examples(suite,profile,teacher,model):
    source=json.loads((suite/"manifest.json").read_bytes())
    prof=json.loads((profile/"manifest.json").read_bytes())
    orig=json.loads((teacher/"manifest.json").read_bytes())
    if prof["model"]!=digest(model) or orig["model"]!=REFERENCE_SHA:
        raise ValueError("model or teacher fingerprint differs")
    if prof["suite"]!=digest(suite/"manifest.json") or prof["suite"]!=orig["suite"] or prof["executable"]!=orig["executable"]:
        raise ValueError("training fixtures or native profiler differ")
    batches=[]
    for i,(win,p,t) in enumerate(zip(source["windows"],prof["files"],orig["files"],strict=True)):
        assert win["file"].split(".")[0]==p["file"].split(".")[0]==t["file"].split(".")[0]
        f=suite/win["file"]
        if digest(f)!=win["sha256"] or digest(profile/p["file"])!=p["sha256"] or digest(teacher/t["file"])!=t["sha256"]:
            raise ValueError("training bytes differ")
        raw=f.read_bytes()
        if not raw.startswith(MAGIC): raise ValueError("wrong fixture type")
        rows=np.frombuffer(raw,dtype=ROW,offset=len(MAGIC))
        scored=rows[rows["score"]==1]
        n=len(scored)
        ids=np.linspace(0,n-1,min(n,1500),dtype=int)
        read=lambda path: np.memmap(path,mode="r",dtype="<f4",shape=(n,400))[ids]
        features=read(profile/p["file"])
        targets=features[:,192:] if profile==teacher else read(teacher/t["file"])[:,192:]
        prior=scored["prior"][ids].view("<f2").astype(np.float32)
        context=np.maximum(np.log(np.maximum(prior,1e-30)),-12)
        context-=context.mean(axis=1,keepdims=True)
        batches.append((features[:,:192],context,scored["target"][ids],features[:,192:],
                        targets,np.full(len(ids),i%7==0,dtype=bool)))
    return tuple(np.concatenate(parts) for parts in zip(*batches))


def probabilities(x, codes, context, u, scales, plm, activation):
    raw=(codes@u.T)*(activation*scales)
    cap=np.tanh(raw/15.)
    gate=x@plm+.5413248546129181
    logits=15*cap+np.logaddexp(0.,gate)[:,None]*context
    exp=np.exp(logits-logits.max(axis=1,keepdims=True))
    return exp/exp.sum(axis=1,keepdims=True),raw,cap,gate


def train(a):
    if digest(BASELINE)!=REFERENCE_SHA: raise ValueError("reference model changed")
    train_manifest=json.loads((a.suite/"manifest.json").read_bytes())
    screen_manifest=json.loads((a.screen_suite/"manifest.json").read_bytes())
    if train_manifest["selection"]==screen_manifest["selection"] and (
        {w["document"] for w in train_manifest["windows"]} & {w["document"] for w in screen_manifest["windows"]}):
        raise ValueError("training and screening documents overlap")
    x,context,y,native,teacher,held=examples(a.suite,a.profile,a.teacher,a.model)
    bank=np2.read_container(str(a.model))
    tensors={t.name:t for t in bank}
    un="unembedding.weight.q"; sn="unembedding.weight.scale"; pn="logit_mixing_head.weight"
    uq=tensors[un].a.astype(np.float32); base_s=bf16(tensors[sn].a).reshape(-1)
    base_p=tensors[pn].a.astype(np.float32).reshape(-1)
    activation=bf16(tensors["unembedding.quantize_activation.scale"].a).item()
    codes=np.clip(np.rint(x/activation),-128,127)
    valid=np.flatnonzero(held)[::2]; training=np.flatnonzero(~held)
    probe=valid[:256]
    probe_p=probabilities(x[probe],codes[probe],context[probe],uq,base_s,base_p,activation)[0]
    mismatch=float(np.max(np.abs(probe_p-native[probe])))
    if mismatch>2e-4: raise ValueError(f"native output-head mismatch {mismatch}")
    provenance=hashlib.sha256((a.profile/"manifest.json").read_bytes()+(a.teacher/"manifest.json").read_bytes()+
                            (a.screen_suite/"manifest.json").read_bytes()+a.replay.read_bytes()+Path(__file__).read_bytes()).hexdigest()
    screen=a.output/"screen"; screen.mkdir(exist_ok=True,parents=True)
    for hard,learn_codes in ((1.,False),(.25,False),(.25,True),(1.,True)):
        spec=dict(kind="head-fit",source_sha256=digest(a.model),label_fraction=hard,
                  fit_integer_codes=learn_codes,steps=a.steps)
        key=hashlib.sha256(json.dumps([provenance,spec],sort_keys=True).encode()).hexdigest()[:20]
        if (screen/(key+".json")).exists(): continue
        rng=np.random.default_rng(10873)
        weights=Adam(uq); logs=Adam(np.zeros_like(base_s)); mix=Adam(base_p)
        def state():
            return (np.rint(weights.value),bf16(round_bf16(base_s*np.exp(logs.value))),mix.value.copy())
        def loss(st):
            total=0.
            for start in range(0,len(valid),512):
                j=valid[start:start+512]
                pred=probabilities(x[j],codes[j],context[j],*st,activation)[0]
                total-=np.log2(np.maximum(pred[np.arange(len(j)),y[j]],3e-8)).sum(dtype=np.float64)
            return total/len(valid)
        best=state(); original=best_loss=loss(best); best_step=0
        for step in range(1,a.steps+1):
            j=rng.choice(training,256,replace=True)
            w,s,g=state()
            pred,raw,cap,gate=probabilities(x[j],codes[j],context[j],w,s,g,activation)
            grad=pred-(1-hard)*teacher[j]
            grad[np.arange(len(j)),y[j]]-=hard
            grad/=len(j)
            dl=grad*(1-cap**2)
            mult=.15+.85*(1+np.cos(np.pi*step/a.steps))/2
            if learn_codes:
                dw=((dl*(s*activation)).T @ codes[j]) + .0001*(weights.value-uq)
                weights.step(dw,step,.012*mult,np.maximum(uq-1.25,-7),np.minimum(uq+1.25,7))
            logs.step((dl*raw).sum(axis=0)+.015*logs.value,step,.001*mult,-.1,.1)
            dg=(grad*context[j]).sum(axis=1)/(1+np.exp(-np.clip(gate,-50,50)))
            mix.step(x[j].T@dg+.01*(mix.value-base_p),step,.00025*mult,base_p-.025,base_p+.025)
            if step in (8,32) or step%100==0 or step==a.steps:
                cur=state(); score=loss(cur)
                if score<best_loss: best_loss,best_step,best=score,step,cur
        print("head",key,"native_max_error",mismatch,"validation_bpt",original,best_loss,"step",best_step,flush=True)
        if not best_step: continue
        w,s,g=best
        replace={un:w.astype(np.int8),sn:round_bf16(s).reshape(tensors[sn].a.shape),pn:g.reshape(tensors[pn].a.shape)}
        new=[np2.Ent(t.name,t.code,replace.get(t.name,t.a)) for t in bank]
        blob,_,expected=np2.encode_np(new,True,np2.np_groups(new,True,True),True)
        with tempfile.TemporaryDirectory(dir=screen) as td:
            dest=Path(td); model=dest/"model.c5"; model.write_bytes(blob)
            back=np2.read_container(str(model))
            if len(back)!=len(expected) or not all(t.same(v) for t,v in zip(expected,back)): raise ValueError("NP2 mismatch")
            r=measure(a.replay,a.screen_suite,model,a.output/"discovery",key)
            record={k:r[k] for k in ("candidate_sha256","candidate_bytes","twice_weight_delta","search_proxy_bytes",
                                      "confirmation_proxy_bytes","proxy_delta_bytes","bits_per_target_delta_by_window","replay_seconds")}
            record.update(key=key,specs=[spec],provenance=provenance,validation_bpt=[original,best_loss],best_step=best_step,
                          train_targets=len(training),validation_targets=len(valid),native_max_error=mismatch)
            model.replace(screen/(key+".c5")); path=dest/"r.json";path.write_text(json.dumps(record,indent=2)+"\n");path.replace(screen/(key+".json"))
            print("screened",key,r["search_proxy_bytes"],r["confirmation_proxy_bytes"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    c=sub.add_parser("collect")
    for name in ("exe","suite","model","output"): c.add_argument(name,type=Path)
    t=sub.add_parser("train")
    for name in ("replay","screen_suite","suite","profile","teacher","model","output"): t.add_argument(name,type=Path)
    t.add_argument("--steps",type=int,default=800)
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path): setattr(a,k,v.resolve())
    if a.action=="collect": collect(a.exe,a.suite,a.model,a.output)
    else: train(a)
