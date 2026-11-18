"""Report the sensitivity of a measured quantization proxy to its sampled articles."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate import FULL_TARGETS


def summarize(trial, manifest_path, transfer):
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=trial["suite_sha256"]:raise ValueError("different fixture")
    manifest=json.loads(raw)
    ref=trial["reference"];candidate=trial["candidate"]
    if len({w["document"] for w in manifest["windows"]})!=len(ref):raise ValueError("duplicate or missing articles")
    for w,r,c in zip(manifest["windows"],ref,candidate,strict=True):
        if w["targets"]!=r["targets"] or r["targets"]!=c["targets"]:raise ValueError("target counts differ")
    bits=np.array([c["bits"]-r["bits"] for r,c in zip(ref,candidate,strict=True)])
    targets=np.array([r["targets"] for r in ref])
    delta=bits/targets
    if not np.allclose(delta,trial["bits_per_target_delta_by_window"],rtol=1e-10,atol=1e-12):
        raise ValueError("inconsistent paired loss")
    # Each document supplies one paired observation. Resample within adjacent
    # chronological groups to preserve broad coverage of this systematic sample.
    rng=np.random.default_rng(93171)
    strata=np.array_split(delta,max(1,len(delta)//8))
    resampled=sum(rng.choice(xs,(10000,len(xs)),replace=True).sum(axis=1) for xs in strata)/len(delta)
    scale=transfer*FULL_TARGETS/8
    weight=trial["twice_weight_delta"]
    percentiles=np.quantile(weight+scale*resampled,[.025,.975])
    return dict(candidate_sha256=trial["candidate_sha256"],suite_sha256=trial["suite_sha256"],
                documents=len(delta),targets=int(targets.sum()),improved_documents=int((delta<0).sum()),
                tied_documents=int((delta==0).sum()),bits_per_target_equal_document=float(delta.mean()),
                bits_per_target_token_weighted=float(bits.sum()/targets.sum()),weight_delta_both_copies=weight,
                measured_transfer_assumption=transfer,projected_entry_delta_bytes=float(weight+scale*delta.mean()),
                article_resampling_95_percentile_bytes=percentiles.tolist())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("trial",type=Path);p.add_argument("manifest",type=Path)
    p.add_argument("--transfer",type=float,default=1.)
    a=p.parse_args()
    print(json.dumps(summarize(json.loads(a.trial.read_bytes()),a.manifest,a.transfer),indent=2))
