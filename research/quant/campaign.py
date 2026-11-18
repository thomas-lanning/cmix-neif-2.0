"""Adaptive fixed-geometry search, with separate fresh-document validation."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

from allocation import export, np2
from evaluate import BASELINE, measure


def recipe(family, layers, fraction=0.25, levels=3):
    return dict(kind="rows", family=family, layers=list(layers), fraction=fraction, levels=levels)


def seeds():
    families = ("down", "up", "attention")
    for family in families:
        for first in (0, 4, 8, 12):
            yield [recipe(family, range(first, first + 4))]
        for mantissa in (5, 3):
            yield [dict(kind="scale", family=family, layers=list(range(16)), mantissa=mantissa)]
        for fraction in (0.0625, 0.25, 1.0):
            yield [recipe(family, range(16), fraction)]


def search(args):
    bank = np2.read_container(str(BASELINE))
    path = args.output / "screen"
    path.mkdir(exist_ok=True)
    identity = hashlib.sha256((Path(__file__).with_name("allocation.py")).read_bytes()).hexdigest()
    suite_hash = hashlib.sha256((args.suite / "manifest.json").read_bytes()).hexdigest()
    provenance = [identity, suite_hash, hashlib.sha256(args.replay.read_bytes()).hexdigest(),
                  hashlib.sha256(Path(__file__).with_name("evaluate.py").read_bytes()).hexdigest()]
    def evaluate_case(specs):
        key = hashlib.sha256(json.dumps([provenance, specs], sort_keys=True).encode()).hexdigest()[:20]
        result_file = path / (key + ".json")
        if result_file.exists():
            return json.loads(result_file.read_bytes())
        if time.time() >= args.until:
            raise TimeoutError("campaign budget reached")
        print("testing", key, json.dumps(specs), flush=True)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(dir=path) as td:
            tmp = Path(td)
            model = tmp / "export.c5"
            counts = export(bank, specs, model)
            r = measure(args.replay, args.suite, model, args.output / "discovery", key)
            compact = {k: r[k] for k in ("candidate_sha256", "candidate_bytes", "twice_weight_delta",
                       "search_proxy_bytes", "confirmation_proxy_bytes", "proxy_delta_bytes",
                       "bits_per_target_delta_by_window", "replay_seconds")}
            compact.update(key=key, specs=specs, changed=counts, provenance=provenance,
                           total_seconds=round(time.monotonic()-started, 3))
            model.replace(path / (key + ".c5"))
            record = tmp / "result.json"
            record.write_text(json.dumps(compact, indent=2)+"\n")
            record.replace(result_file)
        print("screened", key, compact["search_proxy_bytes"], compact["confirmation_proxy_bytes"], flush=True)
        return compact

    def run(specs):
        try:
            return evaluate_case(specs)
        except (ValueError, FloatingPointError, subprocess.SubprocessError, SystemExit) as exc:
            with (path / "failures.jsonl").open("a") as f:
                print(json.dumps(dict(specs=specs, error=str(exc), time=time.time())), file=f)
            print("failed", specs, str(exc), flush=True)
            return None

    first = [r for plan in seeds() if (r := run(plan)) is not None]
    # Vary strength and localize the four most promising families using discovery only.
    ranked = sorted((r for r in first if r["specs"][0]["kind"] == "rows" and
                     len(r["specs"][0]["layers"]) == 4), key=lambda r: r["search_proxy_bytes"])
    second = []
    for r in ranked[:4]:
        s = r["specs"][0]
        for fraction, levels in ((0.0625, 3), (0.5, 3), (1, 3), (0.125, 1), (0.5, 1)):
            second.append(run([recipe(s["family"], s["layers"], fraction, levels)]))
        for layer in s["layers"]:
            second.append(run([recipe(s["family"], [layer], 0.5)]))
    # Search for additive changes on distinct tensor groups.
    proposals = sorted((r for r in first + second if r), key=lambda r: r["search_proxy_bytes"])
    combined = []
    covered = set()
    for r in proposals:
        if r["search_proxy_bytes"] >= 0:
            break
        s = r["specs"][0]
        sites = {(s["family"], layer) for layer in s["layers"]}
        if sites & covered:
            continue
        trial = run(combined + [s]) if combined else r
        if trial and trial["search_proxy_bytes"] < 0 and (not combined or trial["search_proxy_bytes"] < best):
            combined.append(s)
            covered.update(sites)
            best = trial["search_proxy_bytes"]
        if len(combined) >= 6:
            break


def validate(args):
    screen = args.output / "screen"
    candidates = [json.loads(p.read_bytes()) for p in screen.glob("*.json")]
    candidates.sort(key=lambda r: max(r["search_proxy_bytes"], r["confirmation_proxy_bytes"]))
    # Reserve one slot for the best discovery-only candidate as a calibration control.
    if not candidates:
        raise ValueError("no completed screen")
    take = candidates[:args.limit-1] + [min(candidates, key=lambda r: r["search_proxy_bytes"])]
    take = list({r["key"]: r for r in take}.values())
    for r in take:
        if time.time() >= args.until:
            raise TimeoutError("validation budget reached")
        key = r["key"]
        directory = args.output / ("validation-" + hashlib.sha256((args.suite / "manifest.json").read_bytes()).hexdigest()[:12])
        directory.mkdir(exist_ok=True)
        verdict = directory / (key + ".json")
        if verdict.exists():
            continue
        model = screen / (key + ".c5")
        if hashlib.sha256(model.read_bytes()).hexdigest() != r["candidate_sha256"]:
            raise ValueError("candidate differs from screened artifact")
        result = measure(args.replay, args.suite, model, directory, key)
        tmp = verdict.with_suffix(".tmp")
        tmp.write_text(json.dumps(result)+"\n")
        tmp.replace(verdict)
        print("validated", key, result["search_proxy_bytes"], result["confirmation_proxy_bytes"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["search", "validate"])
    p.add_argument("replay", type=Path)
    p.add_argument("suite", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--until", type=float, default=math.inf, help="UTC deadline as Unix time")
    p.add_argument("--limit", type=int, default=6, choices=range(2, 33))
    args = p.parse_args()
    args.replay, args.suite, args.output = (p.resolve() for p in (args.replay, args.suite, args.output))
    args.output.mkdir(exist_ok=True, parents=True)
    lockdir = args.output / ("lock-" + args.phase + "-" + hashlib.sha256(str(args.suite).encode()).hexdigest()[:12])
    with lockdir.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (search if args.phase == "search" else validate)(args)


if __name__ == "__main__":
    main()
