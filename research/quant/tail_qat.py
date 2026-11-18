"""Fit quantized final-MLP weights through the depth mixture and actual token objective."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from allocation import bf16, np2, reduce_rows, round_bf16
from distill import Adam
from evaluate import BASELINE, REFERENCE_SHA, measure
from fixtures import ROW as FIXTURE, MAGIC
from head_fit import probabilities
from tail_profile import ROW, sha


def examples(suite, tail, head):
    manifest = json.loads((suite/"manifest.json").read_bytes())
    profiles = [json.loads((p/"manifest.json").read_bytes()) for p in (tail, head)]
    for p in profiles:
        if p["model"] != REFERENCE_SHA or p["suite"] != sha(suite/"manifest.json"):
            raise ValueError("training profile provenance differs")
    data = []
    for i, (w,t,h) in enumerate(zip(manifest["windows"], profiles[0]["files"], profiles[1]["files"], strict=True)):
        source = suite/w["file"]
        native = tail/t["file"]
        teacher = head/h["file"]
        if (sha(source),sha(native),sha(teacher)) != (w["sha256"],t["sha256"],h["sha256"]):
            raise ValueError("training inputs changed")
        if t["document"] != w["document"] or teacher.stem != source.stem:
            raise ValueError("training profiles are out of order")
        n = w["targets"]
        idx = np.linspace(0, n-1, min(n,768), dtype=int)
        coded = np.memmap(native,mode="r",dtype=ROW,shape=(n,))[idx]
        ref = np.memmap(teacher,mode="r",dtype="<f4",shape=(n,400))[idx]
        raw = source.read_bytes()
        if not raw.startswith(MAGIC): raise ValueError("invalid fixture")
        rows = np.frombuffer(raw,dtype=FIXTURE,offset=len(MAGIC))
        scored = rows[rows["score"]==1][idx]
        prior = scored["prior"].view("<f2").astype(np.float32)
        context = np.maximum(np.log(np.maximum(prior,1e-30)),-12)
        context -= context.mean(axis=1,keepdims=True)
        # The frozen sources can be collapsed to their weighted mean and the
        # log of their total mass. Only the running partial depends on this MLP.
        mean = coded["sum"]/coded["normalizer"][:,None]
        mass = coded["max_logit"]+np.log(coded["normalizer"])
        data.append((coded["down"],coded["partial"],mean,mass,context,scored["target"],ref,
                     np.full(len(idx),i%7==0)))
    return tuple(np.concatenate(fields) for fields in zip(*data))


def normalized(x, epsilon):
    inv = 1/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+epsilon)
    return x*inv,inv


def run(a):
    if sha(BASELINE)!=REFERENCE_SHA: raise ValueError("published weights changed")
    sets = [json.loads((p/"manifest.json").read_bytes())["windows"] for p in (a.training,a.screen)]
    if {w["document"] for w in sets[0]} & {w["document"] for w in sets[1]}:
        raise ValueError("training and screening documents overlap")
    h,partial,completed,log_mass,context,y,ref,held = examples(a.training,a.tail,a.head)
    bank = np2.read_container(str(BASELINE))
    tensors = {t.name:t.a for t in bank}
    key = "blocks.15.mlp.down.weight."
    q0 = tensors[key+"q"]
    s0 = tensors[key+"scale"].reshape(-1)
    query = tensors["blocks.15.q_final.w"].reshape(-1)
    down_scale = bf16(tensors["blocks.15.mlp.down.quantize_activation.scale"]).item()
    out_scale = bf16(tensors["unembedding.quantize_activation.scale"]).item()
    out_q = tensors["unembedding.weight.q"].astype(np.float32)
    out_s = bf16(tensors["unembedding.weight.scale"]).reshape(-1)
    mix = tensors["logit_mixing_head.weight"].reshape(-1)
    valid = np.flatnonzero(held)[::2]
    train = np.flatnonzero(~held)
    def forward(idx, q, scale):
        codes = h[idx].astype(np.float32)
        down = (codes @ q.T) * (scale*down_scale)
        part = partial[idx]+down
        pn,pr = normalized(part,1e-6)
        logit = pn@query
        weight = (1/(1+np.exp(np.clip(log_mass[idx]-logit,-40,40))))[:,None]
        diff = part-completed[idx]
        x = completed[idx]+weight*diff
        xn,xr = normalized(x,np.finfo(np.float32).eps)
        quant = np.clip(np.rint(xn/out_scale),-128,127)
        p,_,cap,gate = probabilities(xn,quant,context[idx],out_q,out_s,mix,out_scale)
        return p,(codes,down,pn,pr,logit,weight,diff,xn,xr,cap,gate)
    baseline_p, oracle = forward(valid[:1024],q0.astype(np.float32),bf16(s0))
    native_hidden = float(np.max(np.abs(oracle[7]-ref[valid[:1024],:192])))
    native_prob = np.abs(baseline_p-ref[valid[:1024],192:])
    # Rare changes at an int8 rounding boundary can move individual logits.
    # Check the continuous hidden state as well as the categorical output.
    if native_hidden>1e-4 or np.mean(native_prob)>2e-6 or np.max(native_prob)>1e-2:
        raise ValueError(f"native tail mismatch: hidden {native_hidden}, probabilities {native_prob.max()}, mean {native_prob.mean()}")
    print("native tail errors",native_hidden,native_prob.max(),native_prob.mean(),flush=True)
    evidence = Path(__file__).read_bytes()+a.replay.read_bytes()
    for folder in (a.training,a.screen,a.tail,a.head):evidence += (folder/"manifest.json").read_bytes()
    provenance = hashlib.sha256(evidence).hexdigest()
    def validation(state):
        bits=0.
        for st in range(0,len(valid),256):
            j=valid[st:st+256]
            p,_=forward(j,*state)
            bits-=np.log2(np.maximum(p[np.arange(len(j)),y[j]],3e-8)).sum(dtype=np.float64)
        return bits/len(valid)
    original=validation((q0.astype(np.float32),bf16(s0)))
    screen=a.output/"screen";screen.mkdir(parents=True,exist_ok=True)
    for fraction,limit in ((.125,3),(.5,3),(1.,3),(.25,1),(1.,5)):
        for hard in (.25,1.):
            spec=dict(kind="token-loss-qat",tensor=key+"q",fraction=fraction,limit=limit,
                      label_weight=hard,steps=a.steps)
            tag=hashlib.sha256(json.dumps([provenance,spec],sort_keys=True).encode()).hexdigest()[:20]
            dest=screen/(tag+".json")
            if dest.exists():continue
            rng=np.random.default_rng(11629)
            q=q0.copy(); scale=s0.copy()
            reduce_rows(q,scale,fraction,limit)
            changed=np.any(q!=q0,axis=1)
            bound=np.where(changed,limit,7)[:,None]
            weights=Adam(q); logs=Adam(np.zeros(len(scale)))
            scale=bf16(scale)
            def state():
                return np.rint(weights.value),bf16(round_bf16(scale*np.exp(logs.value)))
            best=state(); start_loss=best_loss=validation(best); best_at=0
            for step in range(1,a.steps+1):
                j=rng.choice(train,128,replace=True)
                qj,sj=state()
                p,ctx=forward(j,qj,sj)
                codes,down,pn,pr,logit,w,diff,xn,xr,cap,gate=ctx
                grad=p-(1-hard)*ref[j,192:]
                grad[np.arange(len(j)),y[j]]-=hard
                grad/=len(j)
                gx=((grad*(1-cap*cap)*out_s)@out_q)*(np.abs(xn/out_scale)<127.5)
                gg=(grad*context[j]).sum(axis=1)/(1+np.exp(-np.clip(gate,-40,40)))
                gx+=gg[:,None]*mix
                gx=xr*(gx-xn*(gx*xn).mean(axis=1,keepdims=True))
                gl=(gx*diff).sum(axis=1,keepdims=True)*w*(1-w)
                gp=gx*w+gl*pr*(query-pn*(logit[:,None]/192))
                decay=.15+.85*(1+np.cos(np.pi*step/a.steps))/2
                dq=((gp*(sj*down_scale)).T @ codes)+.00006*(weights.value-q)
                weights.step(dq,step,.018*decay,np.maximum(-bound,q.astype(np.float32)-1.3),
                             np.minimum(bound,q.astype(np.float32)+1.3))
                logs.step((gp*down).sum(axis=0)+.002*logs.value,step,.0015*decay,-.2,.2)
                if step in (16,64) or step%200==0 or step==a.steps:
                    st=state(); cost=validation(st)
                    if cost<best_loss:best_loss,best_at,best=cost,step,st
            qfit,sfit=best
            updates={key+"q":qfit.astype("i1"), key+"scale":round_bf16(sfit).reshape(tensors[key+"scale"].shape)}
            new=[np2.Ent(t.name,t.code,updates.get(t.name,t.a)) for t in bank]
            blob,_,permuted=np2.encode_np(new,True,np2.np_groups(new,True,True),True)
            with tempfile.TemporaryDirectory(dir=screen) as d:
                tmp=Path(d); model=tmp/"model.c5";model.write_bytes(blob)
                decoded=np2.read_container(str(model))
                if not all(t.same(u) for t,u in zip(permuted,decoded,strict=True)):raise ValueError("NP2 round trip failed")
                r=measure(a.replay,a.screen,model,a.output/"discovery",tag)
                row={k:r[k] for k in ("candidate_bytes","candidate_sha256","twice_weight_delta","search_proxy_bytes",
                                       "confirmation_proxy_bytes","proxy_delta_bytes","bits_per_target_delta_by_window")}
                row.update(key=tag,specs=[spec],provenance=provenance,best_step=best_at,
                           training_targets=len(train),validation_targets=len(valid),
                           validation_bpt=dict(published=original,initial=start_loss,fitted=best_loss),
                           native_hidden_error=native_hidden,native_probability_error=float(native_prob.max()))
                model.replace(screen/(tag+".c5"))
                log=tmp/"result.json";log.write_text(json.dumps(row,indent=2)+"\n");log.replace(dest)
                print("tail",tag,original,start_loss,best_loss,best_at,r["search_proxy_bytes"],r["confirmation_proxy_bytes"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("replay","screen","training","tail","head","output"):p.add_argument(name,type=Path)
    p.add_argument("--steps",type=int,default=1000)
    a=p.parse_args()
    for name in ("replay","screen","training","tail","head","output"):setattr(a,name,getattr(a,name).resolve())
    run(a)
