"""Price alternative signs for quantized rows with identical effective weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from allocation import bf16, np2
from evaluate import BASELINE, REFERENCE_SHA, measure


def orientation(q, method, threshold):
    counts=np.stack([np.count_nonzero(q==i,axis=1) for i in range(-7,8)],axis=1)
    signal=np.sign(q).mean(axis=1)
    if method=="magnitude":
        signal=q.mean(axis=1)/np.maximum(np.abs(q).max(axis=1),1)
    flip=signal < -threshold
    if method=="fit":
        for _ in range(12):
            hist=np.where(flip[:,None],counts[:,::-1],counts).sum(axis=0)+1
            # Charge one sign per BF16 row scale according to its observed frequency.
            bitcost=-np.log2(hist/hist.sum())
            odds=(flip.sum()+1)/((~flip).sum()+1)
            proposal=(counts[:,::-1]-counts)@bitcost < np.log2(odds)
            if np.array_equal(flip,proposal): break
            flip=proposal
    return flip


def transform(tensors,group,method,threshold):
    names={t.name:t for t in tensors}
    groups=np2.np_groups(tensors,True,True)
    keys={tensors[g.key].name for g in groups}
    changed=0
    for t in tensors:
        if not t.name.endswith(".weight.q") or t.name in keys or t.code!=0: continue
        if group=="embeddings" and "embedding" not in t.name: continue
        if group=="input" and (not "embedding" in t.name or t.name.startswith("unembedding")): continue
        if group=="tables" and not any(k in t.name for k in ("embedding.weight","_table.weight","value_embeddings")): continue
        if group=="prior" and not t.name.startswith("prior_embedding"): continue
        if group=="attention" and ".attention." not in t.name: continue
        pair=names.get(t.name[:-1]+"scale")
        if pair is None or pair.code!=1 or pair.a.size!=t.rows(): continue
        q=t.m2(); s=pair.a.reshape(-1)
        if np.abs(q).max()>7: continue
        mask=orientation(q,method,threshold)
        if not np.any(mask): continue
        old=q[mask]*bf16(s[mask])[:,None]
        q[mask]*=-1
        s[mask]^=np.uint16(32768)
        new=q[mask]*bf16(s[mask])[:,None]
        if not np.array_equal(old,new):
            raise ValueError(f"effective row values changed in {t.name}")
        changed+=np.count_nonzero(mask)
    return int(changed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("replay","suite","output"): p.add_argument(name,type=Path)
    a=p.parse_args()
    a.replay,a.suite,a.output=(p.resolve() for p in (a.replay,a.suite,a.output))
    if hashlib.sha256(BASELINE.read_bytes()).hexdigest()!=REFERENCE_SHA: raise ValueError("published weights changed")
    base=np2.read_container(str(BASELINE))
    provenance=hashlib.sha256(Path(__file__).read_bytes()+a.replay.read_bytes()+(a.suite/"manifest.json").read_bytes()).hexdigest()
    screen=a.output/"screen";screen.mkdir(exist_ok=True,parents=True)
    for method,threshold in (("sign",0.),("fit",.06),("magnitude",0.),("sign",.12),("sign",.24)):
        for group in ("tables","input","embeddings","prior","attention","all"):
            spec=dict(kind="equivalent-row-sign",group=group,method=method,threshold=threshold)
            key=hashlib.sha256(json.dumps([provenance,spec],sort_keys=True).encode()).hexdigest()[:20]
            if (screen/(key+".json")).exists(): continue
            start=time.monotonic()
            bank=[np2.Ent(t.name,t.code,t.a) for t in base]
            count=transform(bank,group,method,threshold)
            if not count: continue
            blob,_,permuted=np2.encode_np(bank,True,np2.np_groups(bank,True,True),True)
            ordered=all(t.same(u) for t,u in zip(bank,permuted,strict=True))
            delta=len(blob)-BASELINE.stat().st_size
            print("packed",key,spec,"rows",count,"delta",delta,"same_neuron_order",ordered,flush=True)
            if delta>=0 or not ordered: continue
            with tempfile.TemporaryDirectory(dir=screen) as td:
                tmp=Path(td); model=tmp/"model.c5"; model.write_bytes(blob)
                back=np2.read_container(str(model))
                if not all(t.same(u) for t,u in zip(bank,back,strict=True)): raise ValueError("decoded NP2 differs")
                r=measure(a.replay,a.suite,model,a.output/"discovery",key)
                record={k:r[k] for k in ("candidate_sha256","candidate_bytes","twice_weight_delta","search_proxy_bytes",
                    "confirmation_proxy_bytes","proxy_delta_bytes","bits_per_target_delta_by_window","replay_seconds")}
                record.update(key=key,specs=[spec],changed=count,provenance=provenance,unchanged_neuron_order=True,
                              total_seconds=time.monotonic()-start)
                model.replace(screen/(key+".c5"))
                output=tmp/"r.json";output.write_text(json.dumps(record,indent=2)+"\n");output.replace(screen/(key+".json"))
                print("screened",key,r["search_proxy_bytes"],r["confirmation_proxy_bytes"],flush=True)


if __name__=="__main__": main()
