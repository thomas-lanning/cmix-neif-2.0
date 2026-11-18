"""Freeze a cross-campaign shortlist and replay its candidates on fresh documents."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from evaluate import measure
from transfer import adjusted, fit


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def make_plan(args):
    fits = [fit(args.reference, candidate, args.interval, results)
            for candidate, results in args.control]
    # Prefer the largest denominator. Near-zero head deltas make the ratio
    # sensitive even to a single archive byte or a small sampling difference.
    strong = max(fits, key=lambda f: abs(f["estimated_primary_delta"]))
    alpha = strong["transfer"]
    if alpha is None or alpha <= 0:
        alpha = 1.
    selected = {}
    counts = {}
    for directory in args.screens:
        rows = [read(p) for p in directory.glob("*.json")]
        counts[str(directory)] = len(rows)
        # Sample different assumptions to avoid choosing a whole shortlist
        # using one sensitivity estimate. Each candidate can appear only once.
        for scale in (alpha, alpha/2, 1.):
            ranked = sorted(rows, key=lambda r: max(adjusted(r, scale)))
            for row in ranked[:2]:
                key = row["candidate_sha256"]
                selected.setdefault(key, dict(key=row["key"], sha256=key,
                    model=str((directory/(row["key"]+".c5")).resolve()), specs=row["specs"],
                    twice_weight_delta=row["twice_weight_delta"]))
    if not selected:
        raise ValueError("no completed screenings")
    return dict(controls=fits, ranking_transfer=alpha, screened=counts,
                candidates=list(selected.values()))


def assess(args, plan):
    suite_hash = hashlib.sha256((args.suite/"manifest.json").read_bytes()).hexdigest()
    cache = args.output / suite_hash
    cache.mkdir(exist_ok=True)
    rows = []
    for c in plan["candidates"]:
        model = Path(c["model"])
        if hashlib.sha256(model.read_bytes()).hexdigest() != c["sha256"]:
            raise ValueError("candidate bytes differ from frozen shortlist")
        path = cache / (c["sha256"]+".json")
        if not path.exists():
            save(path, measure(args.replay, args.suite, model, cache, c["key"]))
        trial = read(path)
        if trial["suite_sha256"] != suite_hash or trial["candidate_sha256"] != c["sha256"]:
            raise ValueError("cached result does not match the frozen experiment")
        estimates = adjusted(trial, plan["ranking_transfer"])
        rows.append(dict(**c, split_estimates=estimates, average=statistics.mean(estimates),
                         max_split=max(estimates), unscaled=adjusted(trial,1.)))
    rows.sort(key=lambda r: r["max_split"])
    return dict(suite=suite_hash, ranking_transfer=plan["ranking_transfer"], ranked=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("freeze")
    prep.add_argument("--reference", type=Path, required=True)
    prep.add_argument("--interval", type=Path, required=True)
    prep.add_argument("--control", type=Path, nargs=2, action="append", required=True,
                      metavar=("PREFIX", "INTERVAL_RESULTS"))
    prep.add_argument("screens", nargs="+", type=Path)
    test = sub.add_parser("replay")
    test.add_argument("replay", type=Path)
    test.add_argument("suite", type=Path)
    test.add_argument("report", type=Path)
    for cmd in (prep,test):
        cmd.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    plan = args.output/"plan.json"
    if args.action=="freeze":
        if plan.exists():
            raise SystemExit("shortlist already frozen")
        save(plan,make_plan(args))
    else:
        result = assess(args,read(plan))
        save(args.report,result)
        best = result["ranked"][0]
        print(best["key"],best["split_estimates"],flush=True)


if __name__=="__main__":
    main()
