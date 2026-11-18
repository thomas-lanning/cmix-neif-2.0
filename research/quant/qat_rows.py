"""Distill seven-level rows with straight-through quantization through the full model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cpu_fisher import cpu_model, digest, inputs
from evaluate import BASELINE, measure
from fixtures import MAGIC, ROW


def crops(model, suite, independent=False, with_teacher=True):
    manifest = json.loads((suite/"manifest.json").read_bytes())
    train, validation = [], []
    windows = manifest["windows"] if independent else manifest["windows"][::2]
    for i, w in enumerate(windows):
        path = suite/w["file"]
        if digest(path) != w["sha256"]:
            raise ValueError("fixture changed")
        raw = path.read_bytes()
        if not raw.startswith(MAGIC):
            raise ValueError("invalid fixture")
        rows = np.frombuffer(raw,dtype=ROW,offset=len(MAGIC))
        scored = np.flatnonzero(rows["score"])
        fit_article = i % 6 != 5 if independent else i < 4
        split = train if fit_article else validation
        if independent:
            # Sample within each half, away from the warm-up and terminal boundaries.
            n = 2 if fit_article else 1
            edges = np.linspace(0, len(scored)-32, n+1, dtype=int)
            starts = (edges[:-1]+edges[1:])//2
        else:
            starts = np.linspace(0, len(scored)-32, 12 if fit_article else 4, dtype=int)
        for start in starts:
            positions = scored[start:start+32]
            # Both models see the same 128 preceding rows for distillation.
            # Final screening replays the native model from the article reset.
            lo, hi = max(0,int(positions[0])-128), int(positions[-1])+1
            token, prior = inputs(rows[lo:hi])
            index = torch.from_numpy(positions-lo)
            logp = None
            if with_teacher:
                with torch.no_grad():
                    logp = model.compute_logits(token,prior,None)[0,index].log_softmax(-1)
            targets = torch.from_numpy(rows["target"][positions].astype(np.int64))
            split.append((token,prior,index,logp,(w["document"],lo,hi),targets))
    return train, validation


class RowFit(torch.nn.Module):
    def __init__(self, codes, scales, indices):
        super().__init__()
        self.register_buffer("indices",torch.tensor(indices,dtype=torch.long))
        self.register_buffer("base_scale",torch.tensor(scales,dtype=torch.float32))
        self.register_buffer("levels",torch.tensor([0.,2.,4.,7.]))
        self.register_buffer("cuts",torch.tensor([1.,3.,5.5]))
        self.latent = torch.nn.Parameter(torch.tensor(codes,dtype=torch.float32))
        self.log_gain = torch.nn.Parameter(torch.zeros(len(indices)))

    def integer_codes(self):
        return self.latent.sign()*self.levels[torch.bucketize(self.latent.abs(),self.cuts)]

    def scale(self):
        return self.base_scale*self.log_gain.exp()

    def replace(self, module, args, original):
        rounded = self.integer_codes()
        codes = self.latent+(rounded-self.latent).detach()
        scale = self.scale()
        rounded_scale = scale.to(torch.bfloat16).float()
        scale = scale+(rounded_scale-scale).detach()
        return original.index_copy(0,self.indices,codes*scale[:,None])


def install(model, tensors, ranking, count):
    selected = {}
    for _,name,row in ranking[:count]:
        selected.setdefault(name,[]).append(row)
    by_name = {t.name:t for t in tensors}
    fits, handles = {}, []
    for name, rows in selected.items():
        layer = int(name.split(".")[1])
        q = by_name[name].m2()[rows]
        raw = by_name[name[:-1]+"scale"].a.reshape(-1)[rows].astype("uint32")
        s = (raw<<16).view("float32")
        fit = RowFit(q,s,rows)
        handles.append(model.blocks[layer].mlp.down.quantize_weight.register_forward_hook(fit.replace))
        fits[name]=fit
    return fits, handles


def discrepancy(model, batch):
    token,prior,index,teacher = batch[:4]
    student = model.compute_logits(token,prior,None)[0,index].log_softmax(-1)
    return (teacher.exp()*(teacher-student)).sum(-1).mean()


def test_kl(model, samples):
    with torch.no_grad():
        return float(np.mean([float(discrepancy(model,b)) for b in samples]))


def snapshot(fits):
    return {name:(fit.indices.tolist(),fit.integer_codes().detach().numpy().astype(np.int8),
                  fit.scale().detach().to(torch.bfloat16).view(torch.uint16).numpy().copy())
            for name,fit in fits.items()}


def export(bank, state, path, codec):
    tensors = [codec.Ent(t.name,t.code,t.a) for t in bank]
    rows = {t.name:t for t in tensors}
    for name,(idx,q,scale) in state.items():
        rows[name].m2()[idx] = q
        rows[name[:-1]+"scale"].a.reshape(-1)[idx] = scale
    blob,_,permuted = codec.encode_np(tensors,True,codec.np_groups(tensors,True,True),True)
    path.write_bytes(blob)
    if not all(x.same(y) for x,y in zip(permuted,codec.read_container(str(path)),strict=True)):
        raise ValueError("distilled candidate failed tensor round trip")


def training_crops(a,model,with_teacher=True):
    training_suite = a.training_suite or a.suite
    if a.training_suite:
        if not a.validation_manifest:
            raise ValueError("provide the independently pinned validation manifest")
        from validate_exports import QUICK, VALIDATION
        if digest(a.suite/"manifest.json") != QUICK or digest(a.validation_manifest) != VALIDATION:
            raise ValueError("unknown quick or validation fixture")
        from check_training_split import validate_for_qat
        quick = json.loads((a.suite/"manifest.json").read_bytes())
        validate = json.loads(a.validation_manifest.read_bytes())
        validate_for_qat(training_suite/"manifest.json", quick, validate)
    return training_suite, crops(model,training_suite,independent=bool(a.training_suite),with_teacher=with_teacher)


def run(a):
    torch.set_num_threads(2)
    model = cpu_model(a.checkpoint)
    from candidate import np2
    plan = json.loads(a.ranking.read_bytes())
    if plan["suite"]!=digest(a.suite/"manifest.json"):
        raise ValueError("Fisher plan used different articles")
    bank = np2.read_container(str(BASELINE))
    training_suite, (train, validation) = training_crops(a,model)
    a.output.mkdir()
    for count in (32,256):
        fits,handles = install(model,bank,plan["ranked_rows"],count)
        optimizer = torch.optim.Adam([
            dict(params=[v.latent for v in fits.values()],lr=.025),
            dict(params=[v.log_gain for v in fits.values()],lr=.0015)],eps=1e-10)
        initial = best = snapshot(fits)
        initial_loss = best_loss = test_kl(model,validation)
        best_step = 0
        trace = []
        rng = np.random.default_rng(130363)
        for step in range(1,a.steps+1):
            optimizer.zero_grad(set_to_none=True)
            loss = discrepancy(model,train[rng.integers(len(train))])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite distillation objective")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for fit in fits.values():
                    fit.latent.clamp_(-7,7)
                    fit.log_gain.clamp_(-.125,.125)
            if step%16==0 or step==a.steps:
                score=test_kl(model,validation)
                trace.append(dict(step=step,teacher_student_kl=score))
                if score<best_loss:
                    best,best_loss,best_step=snapshot(fits),score,step
                print("fit",count,step,score,best_step,flush=True)
        for handle in handles:
            handle.remove()
        report=dict(rows=count,steps=a.steps,fit_crops=[b[4] for b in train],
                    validation_crops=[b[4] for b in validation],targets_per_crop=32,
                    initial_kl=initial_loss,best_kl=best_loss,best_step=best_step,
                    history=trace,ranking_sha256=digest(a.ranking),
                    fitting_manifest_sha256=digest(training_suite/"manifest.json"),
                    screening_manifest_sha256=digest(a.suite/"manifest.json"))
        (a.output/f"fit-{count}.json").write_text(json.dumps(report,indent=2)+"\n")
        for label, state in [("initial",initial),("distilled",best)]:
            if label=="distilled" and not best_step:
                continue
            path=a.output/f"{label}-{count}.c5"
            export(bank,state,path,np2)
            if count==32 and label=="initial" and digest(path)!=digest(a.ranking.parent/"fisher-32.c5"):
                raise ValueError("initial quantizer differs from the measured Fisher model")
            result=measure(a.replay,a.suite,path,a.output,f"qat-{label}-{count}")
            print(label,count,result["twice_weight_delta"],result["search_proxy_bytes"],
                  result["confirmation_proxy_bytes"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint","ranking","suite","replay","output"):
        p.add_argument(arg,type=lambda s:Path(s).resolve())
    p.add_argument("--steps",type=int,default=96)
    p.add_argument("--training-suite",type=Path)
    p.add_argument("--validation-manifest",type=Path)
    run(p.parse_args())
