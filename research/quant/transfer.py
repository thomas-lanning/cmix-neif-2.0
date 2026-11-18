"""Measure how a primary-head change transfers to actual mixed-coder prefix bytes."""
import argparse
import json
import math
from pathlib import Path
import statistics

from evaluate import FULL_TARGETS


def checkpoints(folder):
    return {int(k):int(v) for k,v in (line.split() for line in (folder/"prefix-sizes.tsv").read_text().splitlines())}


def fit(reference, candidate, suite, results):
    if not (reference/"verified-prefix.txt").is_file():
        raise ValueError("published prefix needs a byte-verified reference")
    with (candidate/"model.sha256").open() as f:
        candidate_sha=f.read().split()[0]
    manifest=json.loads((suite/"manifest.json").read_bytes())
    import hashlib
    suite_sha=hashlib.sha256((suite/"manifest.json").read_bytes()).hexdigest()
    rows=[json.loads(row) for row in (results/"trials.jsonl").read_text().splitlines()]
    trial=next(r for r in rows if r["candidate_sha256"]==candidate_sha and r["suite_sha256"]==suite_sha)
    src_start,src_stop=manifest["source_interval"]
    start,stop=src_start-6,src_stop-6
    ref,alt=checkpoints(reference),checkpoints(candidate)
    if ref.keys()!=alt.keys() or min(ref)!=start or max(ref)!=stop:
        raise ValueError("prefix checkpoints and fixture coordinates differ")
    delta={k:alt[k]-ref[k] for k in ref}
    mixed=delta[stop]-delta[start]
    n=manifest["interval_nonterminal_targets"]
    primary=statistics.mean(trial["bits_per_target_delta_by_window"])*n/8
    windows=trial["bits_per_target_delta_by_window"]
    split_primary=[statistics.mean(half)*n/8 for half in (windows[::2],windows[1::2])]
    alpha=mixed/primary if abs(primary)>=1 else None
    return dict(candidate_sha256=candidate_sha,suite_sha256=suite_sha,stream_interval=[start,stop],
                scored_nonterminal_targets=n,mixed_payload_delta=mixed,estimated_primary_delta=primary,
                transfer=alpha,checkpoint_deltas=delta,windows=len(manifest["windows"]),
                primary_split_estimates=split_primary)


def adjusted(r, alpha):
    d=r["bits_per_target_delta_by_window"]
    weight=r["twice_weight_delta"]
    return [weight+alpha*statistics.mean(part)*FULL_TARGETS/8 for part in (d[::2],d[1::2])]


def choose(directories, alpha, per_campaign=3):
    picked=[]
    for directory in directories:
        group=[]
        for path in directory.glob("*.json"):
            r=json.loads(path.read_bytes())
            a,b=adjusted(r,alpha)
            group.append(dict(key=r["key"],model=str(directory/(r["key"]+".c5")),specs=r["specs"],
                         candidate_sha256=r["candidate_sha256"],twice_weight_delta=r["twice_weight_delta"],
                         adjusted_search=a,adjusted_confirmation=b))
        group.sort(key=lambda r:max(r["adjusted_search"],r["adjusted_confirmation"]))
        picked.extend(group[:per_campaign])
    return sorted(picked,key=lambda r:max(r["adjusted_search"],r["adjusted_confirmation"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("fit")
    for name in ("reference","candidate","suite","results"): f.add_argument(name,type=Path)
    r=sub.add_parser("rank")
    r.add_argument("fit",type=Path)
    r.add_argument("screens",nargs="+",type=Path)
    args=p.parse_args()
    if args.command=="fit": output=fit(args.reference,args.candidate,args.suite,args.results)
    else:
        alpha=json.loads(args.fit.read_bytes())["transfer"]
        if alpha is None or not math.isfinite(alpha):
            raise SystemExit("insufficient primary signal to estimate transfer")
        output=choose(args.screens,alpha)
    print(json.dumps(output,indent=2))
