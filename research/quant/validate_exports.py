"""Replay frozen quick-benchmark qualifiers on the disjoint 48-article suite."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluate import measure
from paired_stats import summarize


QUICK = "77974c6afc23485b60b0a8444b3060e465fcc9ae5b65cf1c75c5a0090b29753b"
VALIDATION = "1f3c8d35380c9762babc40a11710098fc979e5eb37f0d21078ae86e1c3a85713"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(replay, suite, exports, output):
    if digest(suite/"manifest.json") != VALIDATION:
        raise ValueError("wrong follow-up suite")
    journals = list(exports.rglob("trials.jsonl"))
    if len(journals) != 1:
        raise ValueError("expected one original quick-screen journal")
    trials = [json.loads(s) for s in journals[0].read_text().splitlines()]
    if not trials or any(r["suite_sha256"] != QUICK for r in trials):
        raise ValueError("missing original or mixed fixture results")
    frozen = sorted({r["candidate_sha256"] for r in trials
                     if max(r["search_proxy_bytes"], r["confirmation_proxy_bytes"]) < 0})
    models = {digest(p):p for p in exports.rglob("*.c5")}
    if not set(frozen).issubset(models):
        raise ValueError("missing a retained model")
    output.mkdir()
    (output/"frozen.json").write_text(json.dumps(dict(quick_suite=QUICK,journal_sha256=digest(journals[0]),
                                                     candidates=frozen),indent=2)+"\n")
    alpha = json.loads(Path(__file__).with_name("prefix-calibration.json").read_bytes())["ranking_transfer"]
    outcomes = []
    for sha in frozen:
        result = measure(replay, suite, models[sha], output, f"frozen-{sha[:12]}")
        stats = summarize(result, suite/"manifest.json", alpha)
        outcomes.append(stats)
    (output/"summary.json").write_text(json.dumps(dict(validation_suite=VALIDATION,ranking_transfer=alpha,
                                                       quick_trials=len(trials),outcomes=outcomes),indent=2)+"\n")
    print(f"{len(trials)} quick trials, {len(outcomes)} independent replays")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("replay","suite","exports","output"):
        p.add_argument(name, type=lambda s:Path(s).resolve())
    a = p.parse_args()
    run(a.replay, a.suite, a.exports, a.output)
