"""Use native activation covariance to prune neurons and fit smaller row alphabets."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from allocation import bf16, round_bf16, np2
from evaluate import BASELINE, REFERENCE_SHA, measure


def gram(profile, out):
    manifest = json.loads((profile / "manifest.json").read_bytes())
    if manifest["model_sha256"] != REFERENCE_SHA or len(manifest["documents"]) != 6:
        raise ValueError("expected the six original discovery documents")
    out.mkdir()
    for f in manifest["files"]:
        file = profile / f["file"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError(f"activation fingerprint differs for {file}")
        x = np.memmap(file, dtype="u1", mode="r", shape=(f["rows"], f["width"]))
        h = np.zeros((f["width"], f["width"]), dtype=np.float64)
        for start in range(0, len(x), 1024):
            block = x[start:start+1024].astype(np.float64)
            if "-up." in file.name:
                block -= 128.0
            h += block.T @ block
        np.save(out / (file.stem + ".npy"), h / len(x))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def fit_codes(w, h, limit, gain, sweeps=1):
    """Coordinate descent minimizing activation-weighted squared output error."""
    diagonal = np.maximum(h.diagonal(), 1e-10)
    q = np.clip(np.rint(w / gain[:, None]), -limit, limit)
    for pass_no in range(sweeps):
        residual = (w - q * gain[:, None]) @ h
        order = range(q.shape[1]) if pass_no % 2 == 0 else reversed(range(q.shape[1]))
        for col in order:
            step = residual[:, col] / (diagonal[col] * gain)
            new = np.clip(np.rint(q[:, col] + step), -limit, limit)
            delta = new - q[:, col]
            changed = np.flatnonzero(delta)
            if len(changed):
                residual[changed] -= (delta[changed] * gain[changed])[:, None] * h[col]
                q[changed, col] = new[changed]
    return q


def quantize(q, scale, h, fraction, limit):
    source = q.astype(np.float64)
    peak = np.abs(source).max(axis=1)
    eligible = np.flatnonzero(peak > limit)
    if not len(eligible):
        return 0
    w = source[eligible]
    s = bf16(scale[eligible]).astype(np.float64)
    if np.any(s == 0):
        raise ValueError("zero scale in eligible rows")
    gain = peak[eligible] / limit
    base = w @ h
    # Alternate fitting the codes and the per-row multiplier.
    for _ in range(2):
        codes = fit_codes(w, h, limit, gain)
        denom = ((codes @ h) * codes).sum(axis=1)
        gain = np.maximum((base * codes).sum(axis=1) / np.maximum(denom, 1e-12), 1e-5)
    fitted = round_bf16(gain * s)
    gain = bf16(fitted).astype(np.float64) / s
    codes = fit_codes(w, h, limit, gain)
    error = w - codes * gain[:, None]
    cost = ((error @ h) * error).sum(axis=1) * s**2
    take = np.argsort(cost, kind="stable")[:max(1, math.ceil(len(w)*fraction))]
    q[eligible[take]] = codes[take].astype(np.int8)
    scale[eligible[take]] = fitted[take]
    return len(take)


def prune(names, layer, h, fraction, compensate):
    pre = f"blocks.{layer}.mlp."
    down = names[pre+"down.weight.q"]
    w = down.m2().astype(np.float64)
    scale = bf16(names[pre+"down.weight.scale"].a).reshape(-1, 1)
    energy = ((w*scale)**2).sum(axis=0)
    # Conditional input variance ranks units by how hard they are to predict
    # from the other activations. Keep multiples of 16 for the native kernels.
    conditional = 1 / np.linalg.inv(h).diagonal()
    n = max(16, 16 * round(fraction*w.shape[1]/16))
    removed = np.sort(np.argsort(energy * conditional, kind="stable")[:n])
    kept = np.setdiff1d(np.arange(w.shape[1]), removed, assume_unique=True)
    if len(kept) < 128:
        raise ValueError("MLP too narrow for this campaign")
    if compensate:
        hk = h[np.ix_(kept, kept)]
        proj = np.linalg.solve(hk, h[np.ix_(kept, removed)])
        target = w[:,kept] + w[:,removed] @ proj.T
        adjusted = fit_codes(target, hk, 7, np.ones(w.shape[0]))
        down.a = np.ascontiguousarray(adjusted, dtype=np.int8)
    else:
        down.a = np.ascontiguousarray(down.a[:, kept])
    for suffix in ("up.weight.q","up.weight.scale","up.bias"):
        if pre+suffix in names:
            t = names[pre+suffix]
            t.a = np.ascontiguousarray(t.a[kept])
    return len(removed)


def candidates():
    for fraction in (1/32, 1/8, 1/4, 1/2):
        for group in (range(12), range(4), range(4,8), range(8,12)):
            for comp in (False,True):
                yield dict(kind="prune", family="down", layers=list(group), fraction=fraction, compensate=comp)
    for family in ("down","up"):
        for fraction in (0.0625,0.25,0.5,1.0):
            for group in (range(16),range(4),range(4,8),range(8,12),range(12,16)):
                yield dict(kind="quant", family=family, layers=list(group), fraction=fraction, levels=3)


def search(a):
    bank = np2.read_container(str(BASELINE))
    if hashlib.sha256(BASELINE.read_bytes()).hexdigest() != REFERENCE_SHA:
        raise ValueError("reference changed")
    columns = {}
    manifest = (a.gram / "manifest.json").read_bytes()
    evidence = hashlib.sha256(manifest+Path(__file__).read_bytes()+
                  (a.suite/"manifest.json").read_bytes()+a.replay.read_bytes()+
                  Path(__file__).with_name("evaluate.py").read_bytes()).hexdigest()
    for layer in range(16):
        for family in ("up","down"):
            h=np.load(a.gram / f"layer-{layer:02d}-{family}.npy")
            h.flat[::len(h)+1] += max(h.diagonal().mean()*0.02, 1e-5)
            columns[layer,family] = h
    screen = a.output / "screen"
    screen.mkdir(exist_ok=True, parents=True)
    for spec in candidates():
        if time.time() >= a.until:
            break
        key = hashlib.sha256(json.dumps([evidence,spec], sort_keys=True).encode()).hexdigest()[:20]
        target = screen / (key + ".json")
        if target.exists():
            continue
        print("testing", key, spec, flush=True)
        try:
            started = time.monotonic()
            with tempfile.TemporaryDirectory(dir=screen) as folder:
                tmp = Path(folder)
                tensors = [np2.Ent(t.name,t.code,t.a) for t in bank]
                names = {t.name:t for t in tensors}
                counts = []
                for layer in spec["layers"]:
                    family = spec["family"]
                    h=columns[layer, family]
                    if spec["kind"] == "prune":
                        counts.append(prune(names,layer,h,spec["fraction"],spec["compensate"]))
                    else:
                        p=f"blocks.{layer}.mlp.{family}.weight."
                        counts.append(quantize(names[p+"q"].m2(),names[p+"scale"].a.reshape(-1),
                                               h,spec["fraction"],spec["levels"]))
                blob, _, perm = np2.encode_np(tensors,True,np2.np_groups(tensors,True,True),True)
                model = tmp / "model.c5"
                model.write_bytes(blob)
                back = np2.read_container(str(model))
                if len(perm)!=len(back) or not all(t.same(u) for t,u in zip(perm,back)):
                    raise ValueError("packed model differs from intended tensors")
                r=measure(a.replay,a.suite,model,a.output/"discovery",key)
                compact={k:r[k] for k in ("candidate_sha256","candidate_bytes","twice_weight_delta",
                         "search_proxy_bytes","confirmation_proxy_bytes","proxy_delta_bytes",
                         "bits_per_target_delta_by_window","replay_seconds")}
                compact.update(key=key,specs=[spec],changed=counts,provenance=evidence,
                               total_seconds=round(time.monotonic()-started,3))
                model.replace(screen/(key+".c5"))
                record=tmp/"result.json"
                record.write_text(json.dumps(compact,indent=2)+"\n")
                record.replace(target)
            print("screened",key,compact["search_proxy_bytes"],compact["confirmation_proxy_bytes"],flush=True)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, subprocess.SubprocessError, SystemExit) as exc:
            with (screen/"failures.jsonl").open("a") as log:
                print(json.dumps(dict(key=key,specs=[spec],error=str(exc))),file=log)
            print("failed",key,str(exc),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    g=sub.add_parser("gram")
    g.add_argument("profile",type=Path)
    g.add_argument("output",type=Path)
    s=sub.add_parser("search")
    for name in ("replay","suite","gram","output"):
        s.add_argument(name,type=Path)
    s.add_argument("--until",type=float,default=math.inf)
    args=p.parse_args()
    if args.action=="gram":
        gram(args.profile,args.output)
    else:
        args.replay,args.suite,args.gram,args.output=(p.resolve() for p in (args.replay,args.suite,args.gram,args.output))
        search(args)
