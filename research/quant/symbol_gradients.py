"""Price single-code weight changes using real-token gradients and empirical variance."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cpu_fisher import cpu_model, digest
from evaluate import BASELINE, FULL_TARGETS, REFERENCE_SHA, measure
from qat_rows import training_crops


def token_loss(model, batch):
    tokens, prior, positions, _, _, targets = batch
    logits = model.compute_logits(tokens,prior,None)[0,positions]
    return torch.nn.functional.cross_entropy(logits,targets)


def rank(model, tensors, train):
    proposals = []
    for i,block in enumerate(model.blocks):
        name = f"blocks.{i}.mlp.down.weight."
        q = tensors[name+"q"].m2()
        scale = (tensors[name+"scale"].a.reshape(-1).astype("uint32") << 16).view("float32")
        layer = block.mlp.down
        current = layer.quantize_weight(layer.weight,initialize_scales_steps=None,return_ints=True)
        if not np.array_equal(q,current.detach().numpy()):
            raise ValueError(f"trainer and native row codes differ at {name}")
        layer.weight.requires_grad_(True)
        proposals.append(dict(name=name,q=q,scale=scale,weight=layer.weight,
                              total=np.zeros(q.shape),squares=np.zeros(q.shape)))
    for n,batch in enumerate(train,1):
        grads = torch.autograd.grad(token_loss(model,batch),[p["weight"] for p in proposals])
        for p,g in zip(proposals,grads,strict=True):
            gradient = g.detach().numpy()*p["scale"][:,None]
            p["total"] += gradient
            p["squares"] += np.square(gradient,dtype=np.float64)
        if n%32==0:
            print("gradient crops",n,flush=True)
    pieces=[]
    price_per_bit = 2*math.log(2)/FULL_TARGETS
    for layer,p in enumerate(proposals):
        q=p["q"].astype(np.int16)
        mean=p["total"]/len(train)
        second=p["squares"]/len(train)
        stderr=np.sqrt(np.maximum(second-mean**2,0)/len(train))
        counts=np.bincount((q+7).ravel(),minlength=15)+1.
        price=-np.log2(counts/counts.sum())
        move=np.where(mean>0,-1,1)
        new=q+move
        legal=np.abs(new)<=np.abs(q).max(axis=1,keepdims=True)
        saving=price[q+7]-price[np.clip(new+7,0,14)]
        # 32 targets per crop. Half the diagonal empirical-Fisher term estimates
        # local curvature. The storage rate counts both copies of the model.
        cost=move*mean+16*second+.8*stderr-price_per_bit*saving
        eligible=np.flatnonzero(legal & (saving>.125) & (cost<0))
        if not np.isfinite(mean).all() or not np.isfinite(second).all():
            raise ValueError("nonfinite token gradient")
        pieces.append((cost.ravel()[eligible],np.full(len(eligible),layer),eligible,
                       move.ravel()[eligible],saving.ravel()[eligible]))
    cost,layer,coord,step,saving = (np.concatenate(x) for x in zip(*pieces,strict=True))
    order=np.argsort(cost,kind="stable")
    return proposals, (layer[order],coord[order],step[order],saving[order])


def run(a):
    torch.set_num_threads(2)
    if digest(BASELINE)!=REFERENCE_SHA:
        raise ValueError("unknown reference model")
    model=cpu_model(a.checkpoint)
    # The standalone NP2 exporter supplies a minimal pysrc module for its own imports.
    # Initialize the full training package before loading that exporter.
    from candidate import np2
    train_suite,(training,validation)=training_crops(a,model,with_teacher=False)
    bank=np2.read_container(str(BASELINE))
    lookup={t.name:t for t in bank}
    proposals,(layers,coords,steps,savings)=rank(model,lookup,training)
    a.output.mkdir()
    with torch.no_grad():
        original_val=np.array([float(token_loss(model,b)) for b in validation])
    def run_budget(budget):
        if len(layers)<budget:
            raise ValueError("insufficient qualifying rate-reducing edits")
        tensors=[np2.Ent(t.name,t.code,t.a) for t in bank]
        target={t.name:t for t in tensors}
        with torch.no_grad():
            for i,p in enumerate(proposals):
                used=layers[:budget]==i
                candidate=target[p["name"]+"q"].m2()
                candidate.flat[coords[:budget][used]]+=steps[:budget][used].astype(candidate.dtype)
                physical=candidate*p["scale"][:,None]
                p["weight"].copy_(torch.from_numpy(physical))
                layer=model.blocks[i].mlp.down
                rec=layer.quantize_weight(layer.weight,initialize_scales_steps=None,return_ints=True)
                if not np.array_equal(rec.numpy(),candidate):
                    raise ValueError("updated training codes differ from the candidate")
            validation_delta=[float(token_loss(model,b))-x for b,x in zip(validation,original_val,strict=True)]
        data,_,ordered=np2.encode_np(tensors,True,np2.np_groups(tensors,True,True),True)
        path=a.output/f"gradient-symbols-{budget}.c5"
        path.write_bytes(data)
        if not all(x.same(y) for x,y in zip(ordered,np2.read_container(str(path)),strict=True)):
            raise ValueError("candidate failed its tensor round trip")
        result=measure(a.replay,a.suite,path,a.output,f"gradient-symbols-{budget}")
        record=dict(edited_weights=budget,eligible_weights=len(layers),fit_crops=len(training),
                    fit_targets=len(training)*32,model_selection_crops=len(validation),
                    fitting_manifest_sha256=digest(train_suite/"manifest.json"),
                    predicted_single_copy_saving_bits=float(savings[:budget].sum()),
                    selection_mean_nats_delta_by_crop=validation_delta,candidate_sha256=digest(path),
                    twice_weight_delta=result["twice_weight_delta"],
                    quick_search=result["search_proxy_bytes"],quick_confirmation=result["confirmation_proxy_bytes"])
        (a.output/f"gradient-symbols-{budget}.json").write_text(json.dumps(record,indent=2)+"\n")
        print(budget,record["twice_weight_delta"],record["quick_search"],record["quick_confirmation"],flush=True)
        return record["quick_search"]

    discovery = {budget: run_budget(budget) for budget in (1024, 8192, 32768)}
    if a.refine:
        # The confirmation half and the independent articles stay out of this choice.
        best = min(discovery, key=lambda size: (discovery[size], size))
        neighbours = {1024: (256, 4096), 8192: (4096, 16384), 32768: (16384, 65536)}
        extra = [size for size in neighbours[best] if size <= len(layers)]
        plan = dict(discovery_scores=discovery, best_budget=best, next_budgets=extra)
        (a.output/"refinement.json").write_text(json.dumps(plan,indent=2)+"\n")
        for budget in extra:
            run_budget(budget)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","suite","training_suite","validation_manifest","replay","output"):
        p.add_argument(name,type=lambda s:Path(s).resolve())
    p.add_argument("--refine", action="store_true", help="test two edit budgets around the best discovery result")
    run(p.parse_args())
