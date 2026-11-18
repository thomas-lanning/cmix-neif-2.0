"""Use gradients through the released transformer to rank seven-level row exports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch

from evaluate import BASELINE, REFERENCE_SHA, ROOT, measure
from fixtures import MAGIC, ROW


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_model(checkpoint):
    train = ROOT / "training/tf/3_train"
    sys.path.insert(0, str(train/"tree"))
    sys.path.insert(0, os.environ["TFT_TREE"])
    import rx3a
    import pysrc.model as model
    from pysrc import kda_reference as ref

    # The release includes mathematical CPU implementations of the same
    # three operations used by its GPU KDA layers. They support autograd.
    def kda(q, k, v, g, beta, A_log, dt_bias, cu_seqlens, use_qk_l2norm_in_kernel, use_gate_in_kernel):
        if not use_qk_l2norm_in_kernel or not use_gate_in_kernel:
            raise ValueError("unexpected KDA configuration")
        return ref.kda_recurrent_reference(q,k,v,g,beta,A_log,dt_bias,cu_seqlens,dtype=q.dtype), None

    def conv(x, weight, activation, cu_seqlens):
        if activation != "silu":
            raise ValueError("unexpected convolution activation")
        return ref.causal_conv1d_silu_reference(x,weight,cu_seqlens,dtype=x.dtype), None

    def gate(module, x, z):
        return ref.fused_rmsnorm_gated_reference(x,z,module.weight,eps=module.eps,dtype=x.dtype)

    model.chunk_kda = kda
    model.causal_conv1d = conv
    config = json.loads((train/"chain/06_qat.json").read_bytes())
    config["overrides"]["attention_implementation"] = "sdpa"
    config["arch"]["attnres"]["impl"] = "naive"
    raw, _ = rx3a.build_model(config, "fp32", torch.device("cpu"), False)
    rx3a.load_init(raw, str(checkpoint/"strict.tch"), True, torch.device("cpu"), False)
    rx3a.install_mixed(raw, str(checkpoint/"row-ranges.npz"), False)
    for m in raw.modules():
        if isinstance(m, model.FusedRMSNormGated):
            m.forward = MethodType(gate,m)
    raw.eval().requires_grad_(False)
    return raw


def row_proposals(model, bank):
    names = {e.name:e for e in bank}
    proposals = []
    for layer, block in enumerate(model.blocks):
        module = block.mlp.down
        name = f"blocks.{layer}.mlp.down.weight.q"
        original = names[name].m2().copy()
        levels = np.array([0,0,2,2,4,4,7,7], dtype=np.int8)
        changed = np.sign(original)*levels[np.abs(original)]
        integers = module.quantize_weight(module.weight, initialize_scales_steps=None, return_ints=True)
        if not np.array_equal(original, integers.detach().numpy()):
            raise ValueError(f"training weights differ at {name}")
        scale = module.quantize_weight.scale.detach().flatten().float().to(torch.bfloat16).float()
        delta = torch.tensor(changed.astype(np.float32)-original) * scale[:,None]
        hist = np.bincount((original.ravel()+7), minlength=15).astype(np.float64)+1
        price = -np.log2(hist/hist.sum())
        estimated_gain = (price[original+7] - price[changed+7]).sum(1)
        module.weight.requires_grad_(True)
        proposals.append(dict(name=name, weight=module.weight, delta=delta, changed=changed,
                              gain=estimated_gain, damage=np.zeros(len(original))))
    return proposals


def inputs(rows):
    from zctx import zpaket
    shifts = zpaket.shifts(2)
    token = rows["tok"].astype("i8")
    for name, shift in (("p1",shifts["last"]),("p2",shifts["last"]+8),
                        ("mb",shifts["match"]),("mc",shifts["len"])):
        token |= rows[name].astype("i8") << shift
    prior = rows["prior"].view("<f2").astype(np.float32)
    return torch.from_numpy(token)[None], torch.from_numpy(prior)[None]


def rank(model, proposals, suite, profile):
    meta = json.loads((profile/"manifest.json").read_bytes())
    manifest = json.loads((suite/"manifest.json").read_bytes())
    if meta["model"] != REFERENCE_SHA or meta["suite"] != digest(suite/"manifest.json"):
        raise ValueError("wrong native profile or fixture")
    observations = []
    for w, p in list(zip(manifest["windows"],meta["files"],strict=True))[::2]:
        path, fp = suite/w["file"], profile/p["file"]
        if path.stem != fp.stem or digest(path)!=w["sha256"] or digest(fp)!=p["sha256"]:
            raise ValueError("input fingerprint differs")
        data = path.read_bytes()
        if not data.startswith(MAGIC):
            raise ValueError("bad fixture magic")
        rows = np.frombuffer(data,dtype=ROW,offset=len(MAGIC))
        scored = np.flatnonzero(rows["score"])[:64]
        end = int(scored[-1])+1
        token, prior = inputs(rows[:end])
        logits = model.compute_logits(token,prior,None)[0,scored]
        logp = logits.log_softmax(-1)
        prob = logp.detach().exp()
        reference = np.memmap(fp,mode="r",dtype="<f4").reshape(-1,400)[:len(scored),192:]
        native = torch.tensor(reference.copy(),dtype=torch.float64)
        native /= native.sum(1,keepdim=True)
        kl = (native*(native.clamp_min(1e-30).log()-logp.detach().double())).sum(1).mean().item()
        agreement = (native.argmax(1)==prob.argmax(1)).double().mean().item()
        record = dict(document=w["document"],fed=end,scored=len(scored),
                      native_to_training_mean_kl_nats=kl,top1_agreement=agreement)
        print("alignment",record,flush=True)
        if not np.isfinite(kl) or kl > .15:
            raise ValueError("CPU reference diverges materially from the native primary head")
        rng = torch.Generator().manual_seed(104729+w["document"])
        for draw in range(2):
            targets = torch.multinomial(prob,1,generator=rng).squeeze(1)
            loss = -logp[torch.arange(len(scored)),targets].sum()
            grads = torch.autograd.grad(loss,[p["weight"] for p in proposals],retain_graph=draw==0)
            for r, grad in zip(proposals,grads,strict=True):
                shifted = grad.detach()*r["delta"]
                # Shrink the squared directional estimate towards its diagonal
                # approximation, reducing sensitivity to chance cancellation.
                r["damage"] += (.75*shifted.sum(1).square()+.25*shifted.square().sum(1)).numpy()/len(scored)
        observations.append(record)
    rows = []
    for r in proposals:
        for i, (damage,gain) in enumerate(zip(r["damage"],r["gain"],strict=True)):
            if not np.isfinite(damage):
                raise ValueError("nonfinite native-weight sensitivity")
            if gain>0:
                rows.append((float(damage/gain),r["name"],i))
    return sorted(rows),observations


def run(a):
    torch.set_num_threads(2)
    model = cpu_model(a.checkpoint)
    from candidate import np2
    if digest(BASELINE)!=REFERENCE_SHA:
        raise ValueError("release weights changed")
    bank = np2.read_container(str(BASELINE))
    proposals = row_proposals(model,bank)
    selected,observations = rank(model,proposals,a.suite,a.profile)
    a.output.mkdir()
    plan = dict(suite=digest(a.suite/"manifest.json"),checkpoint=digest(a.checkpoint/"strict.tch"),
                discovery=observations,ranked_rows=selected)
    (a.output/"ranking.json").write_text(json.dumps(plan,indent=2)+"\n")
    alternatives = {r["name"]:r["changed"] for r in proposals}
    for count in (8,32,128):
        if len(selected)<count:
            raise ValueError("insufficient qualifying rows")
        candidate = [np2.Ent(t.name,t.code,t.a) for t in bank]
        names = {t.name:t for t in candidate}
        for _, name, row in selected[:count]:
            names[name].m2()[row] = alternatives[name][row]
        blob, _, permuted = np2.encode_np(candidate,True,np2.np_groups(candidate,True,True),True)
        path = a.output/f"fisher-{count}.c5"
        path.write_bytes(blob)
        if not all(x.same(y) for x,y in zip(permuted,np2.read_container(str(path)),strict=True)):
            raise ValueError("export failed its tensor round trip")
        result = measure(a.replay,a.suite,path,a.output,f"cpu-fisher-{count}")
        print(count,result["twice_weight_delta"],result["search_proxy_bytes"],result["confirmation_proxy_bytes"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint","suite","profile","replay","output"):
        parser.add_argument(arg,type=lambda p:Path(p).resolve())
    run(parser.parse_args())
