"""Recover pruned MLPs by matching frozen native block outputs on separate documents."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
from allocation import bf16, np2, round_bf16
from calibrated import fit_codes
from evaluate import BASELINE, REFERENCE_SHA, measure


def hidden(x, u, fold, down_activation_scale):
    pre = (x @ u.T) * fold
    rect = np.maximum(pre, 0)
    units = rect * rect / down_activation_scale
    return pre, units, np.minimum(np.rint(units), 127)


class Adam:
    def __init__(self, value):
        self.value = value.copy().astype(np.float32)
        self.m = np.zeros_like(self.value)
        self.v = np.zeros_like(self.value)

    def step(self, grad, step, lr, lo, hi):
        self.m *= .9
        self.m += .1 * grad
        self.v *= .99
        self.v += .01 * grad**2
        eta = lr * np.sqrt(1-.99**step)/(1-.9**step)
        self.value -= eta*self.m/(np.sqrt(self.v)+1e-8)
        np.clip(self.value,lo,hi,out=self.value)


def raster(folder, manifest, layer, family):
    name=f"layer-{layer:02d}-{family}.u8"
    info=next(f for f in manifest["files"] if f["file"]==name)
    path=folder/name
    with path.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=info["sha256"]:
            raise ValueError(f"activation file changed: {name}")
    return np.memmap(path,mode="r",dtype="u1",shape=(info["rows"],info["width"]))


def train_layer(tensors, folder, manifest, layer, fraction, steps, batch, destination, refit=False):
    prefix=f"blocks.{layer}.mlp."
    q=lambda part:tensors[prefix+part+".weight.q"].a.astype(np.float32)
    s=lambda part:bf16(tensors[prefix+part+".weight.scale"].a).reshape(-1)
    act=lambda part:bf16(tensors[prefix+part+".quantize_activation.scale"].a).item()
    u,d=q("up"),q("down")
    su,sd=s("up"),s("down")
    au,ad=act("up"),act("down")
    X=raster(folder,manifest,layer,"up")
    H=raster(folder,manifest,layer,"down")
    if len(X)!=len(H):
        raise ValueError("different up and down activation counts")
    rng=np.random.default_rng(4049+layer)
    split=7*len(X)//8
    train=rng.choice(split,min(split,24576),replace=False)
    valid=rng.choice(np.arange(split,len(X)),min(len(X)-split,768),replace=False)
    x=X[train].astype(np.float32)-128
    h=H[train].astype(np.float32)
    xv=X[valid].astype(np.float32)-128
    hv=H[valid].astype(np.float32)
    _,_,oracle=hidden(xv,u,su*au,ad)
    if not np.array_equal(oracle,hv):
        raise ValueError(f"layer {layer}: native teacher mismatch in {np.count_nonzero(oracle!=hv)} activations")
    y=(h @ d.T)*(sd*ad)
    yv=(hv @ d.T)*(sd*ad)
    variance=(y*y).mean(axis=0)
    inv_var=1./np.maximum(variance, max(1e-12,variance.mean()*0.02))
    gram=h.T@h / len(h)
    gram.flat[::len(gram)+1] += max(gram.diagonal().mean()*.02,1e-5)
    cost=np.square(d*sd[:,None]).sum(axis=0) / np.linalg.inv(gram).diagonal()
    n=max(16,16*round(d.shape[1]*fraction/16))
    gone=np.argsort(cost,kind="stable")[:n]
    keep=np.setdiff1d(np.arange(d.shape[1]),gone,assume_unique=True)
    hk=gram[np.ix_(keep,keep)]
    correction=np.linalg.solve(hk,gram[np.ix_(keep,gone)])
    up=Adam(u[keep]); down=Adam(d[:,keep]+d[:,gone]@correction.T)
    ups=Adam(np.zeros(len(keep))); downs=Adam(np.zeros(len(sd)))

    def snapshot():
        return (np.rint(np.clip(up.value,-7,7)).astype("i1"),
                np.rint(np.clip(down.value,-7,7)).astype("i1"),
                round_bf16(su[keep]*np.exp(ups.value)),round_bf16(sd*np.exp(downs.value)))

    def valid_loss(state):
        uc,dc,us,ds=state
        _,_,z=hidden(xv,uc.astype(np.float32),bf16(us)*au,ad)
        err=(z @ dc.astype(np.float32).T)*(bf16(ds)*ad)-yv
        return float((err*err*inv_var).mean())

    best=snapshot()
    initial=best_loss=valid_loss(best)
    best_step=0
    if refit:
        # With fixed hidden codes the down projection is a quadratic problem.
        # Solve its continuous optimum once and alternate integer codes with a
        # representable row scale. Measure the actual native-grid error on held-out rows.
        target=down.value.astype(np.float64)
        cov=hk.astype(np.float64)
        gain=np.ones(len(sd),dtype=np.float64)
        for epoch in range(4):
            codes=fit_codes(target,cov,7,gain,sweeps=2)
            ch=codes@cov
            denominator=(ch*codes).sum(axis=1)
            optimum=(ch*target).sum(axis=1)/np.maximum(denominator,1e-12)
            fitted=round_bf16(sd*np.clip(optimum,.75,1.5))
            gain=bf16(fitted)/np.maximum(sd,1e-30)
            gain=np.where(sd==0,1,gain)
            state=(best[0],codes.astype("i1"),best[2],fitted)
            loss=valid_loss(state)
            if loss<best_loss:
                best_loss,best_step,best=loss,-epoch-1,state
        down=Adam(best[1].astype(np.float32))
        ratio=bf16(best[3])/np.maximum(sd,1e-30)
        downs=Adam(np.log(np.where(sd==0,1,ratio)))
    for step in range(1,steps+1):
        i=rng.integers(0,len(x),batch)
        uc,dc,us,ds=snapshot()
        uc,dc=uc.astype(np.float32),dc.astype(np.float32)
        fu,fd=bf16(us)*au,bf16(ds)*ad
        pre,units,z=hidden(x[i],uc,fu,ad)
        pred=(z @ dc.T)*fd
        dy=(pred-y[i])*(inv_var/batch)
        gd=dy*fd
        dz=(gd @ dc)*(2./ad)*np.maximum(pre,0)*(units<127)
        mult=.20+.80*(1+math.cos(math.pi*step/steps))/2
        down.step(gd.T@z,step,.035*mult,-7.49,7.49)
        up.step((dz*fu).T@x[i],step,.025*mult,-7.49,7.49)
        downs.step((dy*pred).sum(axis=0),step,.001*mult,-.35,.35)
        ups.step((dz*pre).sum(axis=0),step,.0005*mult,-.35,.35)
        if step%64==0 or step==steps or (refit and step in (1,4,16,32)):
            state=snapshot()
            loss=valid_loss(state)
            if loss<best_loss:
                best_loss,best_step,best=loss,step,state
    with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as tmp:
        np.savez(tmp,keep=keep,up=best[0],down=best[1],us=best[2],ds=best[3],
                 initial=initial,best_loss=best_loss,best_step=best_step)
    Path(tmp.name).replace(destination)
    print("fit",layer,fraction,round(initial,6),round(best_loss,6),best_step,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("replay","suite","profile","output"): p.add_argument(name,type=Path)
    p.add_argument("--fractions",type=float,nargs="+",default=[.125,.25,.375,.0625])
    p.add_argument("--steps",type=int,default=1024)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--refit",action="store_true",help="fit integer down-projection codes and BF16 scales before gradient updates")
    p.add_argument("--until",type=float,default=math.inf)
    a=p.parse_args()
    a.replay,a.suite,a.profile,a.output=map(Path.resolve,(a.replay,a.suite,a.profile,a.output))
    f=(a.profile/"manifest.json").read_bytes()
    manifest=json.loads(f)
    if manifest["model_sha256"]!=REFERENCE_SHA or len(manifest["documents"])<24:
        raise ValueError("a separate native training profile is required")
    if set(manifest["documents"]) & {x["document"] for x in json.loads((a.suite/"manifest.json").read_bytes())["windows"]}:
        raise ValueError("training and screening documents overlap")
    if any(not 0<v<=.5 for v in a.fractions) or a.steps<1 or a.batch<1:
        raise ValueError("invalid training budget")
    bank=np2.read_container(str(BASELINE))
    names={t.name:t for t in bank}
    provenance=hashlib.sha256(f+Path(__file__).read_bytes()+
            str((a.steps,a.batch,a.refit)).encode()).hexdigest()
    screen=a.output/"screen"
    fitted=a.output/("fits-"+provenance[:12])
    screen.mkdir(exist_ok=True,parents=True)
    fitted.mkdir(exist_ok=True)
    for fraction in a.fractions:
        for layer in range(12):
            if time.time()>=a.until: return
            dest=fitted/f"{fraction}-{layer}.npz"
            if not dest.exists():
                train_layer(names,a.profile,manifest,layer,fraction,a.steps,a.batch,dest,a.refit)
        for layers in (list(range(12)),list(range(4)),list(range(4,8)),list(range(8,12))):
            if time.time()>=a.until: return
            spec=dict(kind="distill",family="mlp",fraction=fraction,layers=layers,steps=a.steps,refit=a.refit)
            key=hashlib.sha256(json.dumps([spec,provenance],sort_keys=True).encode()).hexdigest()[:20]
            if (screen/(key+".json")).exists(): continue
            started=time.monotonic()
            tensors=[np2.Ent(t.name,t.code,t.a) for t in bank]
            current={t.name:t for t in tensors}
            for layer in layers:
                pre=f"blocks.{layer}.mlp."
                with np.load(fitted/f"{fraction}-{layer}.npz") as v:
                    for dst,src in (("up.weight.q","up"),("down.weight.q","down"),("up.weight.scale","us"),("down.weight.scale","ds")):
                        current[pre+dst].a=np.ascontiguousarray(v[src]).reshape(-1) if src in ("us","ds") else np.ascontiguousarray(v[src])
                    if pre+"up.bias" in current:
                        b=current[pre+"up.bias"]
                        b.a=b.a[v["keep"]].copy()
            data,_,expected=np2.encode_np(tensors,True,np2.np_groups(tensors,True,True),True)
            with tempfile.TemporaryDirectory(dir=screen) as td:
                model=Path(td)/"model.c5"
                model.write_bytes(data)
                back=np2.read_container(str(model))
                if len(back)!=len(expected) or not all(t.same(u) for t,u in zip(expected,back)):
                    raise ValueError("distilled model differs after NP2 decode")
                r=measure(a.replay,a.suite,model,a.output/"discovery",key)
                compact={k:r[k] for k in ("candidate_sha256","candidate_bytes","twice_weight_delta","search_proxy_bytes",
                            "confirmation_proxy_bytes","proxy_delta_bytes","bits_per_target_delta_by_window","replay_seconds")}
                compact.update(key=key,specs=[spec],provenance=provenance,total_seconds=time.monotonic()-started)
                model.replace(screen/(key+".c5"))
                record=Path(td)/"result.json"; record.write_text(json.dumps(compact,indent=2)+"\n")
                record.replace(screen/(key+".json"))
            print("screened",key,r["search_proxy_bytes"],r["confirmation_proxy_bytes"],flush=True)


if __name__=="__main__":
    main()
