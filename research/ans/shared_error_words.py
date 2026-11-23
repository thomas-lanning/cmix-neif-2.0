"""Fit a decoder-visible correction for unanimous errors after lowercase spellings."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess

import numpy as np

from mixer_fit import EVENT, TOTAL, arithmetic_size, digest

STARTS = (16_000_000, 20_000_000, 26_000_000, 34_000_000, 42_000_000)
CANONICAL_SHA = "c35d3073eeba70f22a2e08a50202b53eb95466e3e9f1f2702f6f9b3f96fd8718"
PENALTIES = (4.0, 64.0, 512.0)


def load(root, canonical, extractor, out, start):
    receipt = json.loads((root / "verified" / f"{start}.json").read_bytes())
    path = root / "run" / f"span-{start}.bin"
    if (
        digest(path) != receipt["trace_sha256"]
        or path.stat().st_size != EVENT.itemsize * receipt["events"]
    ):
        raise ValueError("trace differs from its verified receipt")
    features = out / f"votes-{start}.bin"
    subprocess.run(
        [extractor, canonical, str(start), str(receipt["end"] - start), path, features], check=True
    )
    ev = np.memmap(path, dtype=EVENT, mode="r")
    votes = np.memmap(features, dtype=np.uint8, mode="r").reshape(len(ev), 2)
    pref = ev["prefix"]
    p = ev["final"].astype(float) / TOTAL
    favored = p >= 0.5
    agree = np.ones(len(ev), dtype=bool)
    for stage in ("floor", "head", "base"):
        agree &= (ev[stage] >= TOTAL // 2) == favored
    # A prefix from 8 through 63 means 3, 4 or 5 earlier bits of the byte are known.
    target = agree & (ev["previous"] >= 97) & (ev["previous"] <= 122) & (pref >= 8) & (pref < 64)
    use = np.flatnonzero(target & (votes[:, 1] >= 2))
    v = votes[use].astype(float)
    odds = np.log(p[use]) - np.log1p(-p[use])
    base = np.clip(odds, -8, 8)
    spelling = np.log((v[:, 0] + 0.5) / (v[:, 1] - v[:, 0] + 0.5))
    x = np.column_stack((np.ones(len(use)), base, (v[:, 1] - 2) / 14, spelling))
    return dict(
        ev=ev,
        use=use,
        x=x,
        odds=odds,
        y=ev["bit"][use].astype(float),
        target_events=int(target.sum()),
        events=len(ev),
        source_sha256=receipt["trace_sha256"],
    )


def fit(d, n_features, penalty):
    x = d["x"][:, :n_features]
    beta = np.zeros(n_features)
    for _ in range(8):
        probability = 1 / (1 + np.exp(-np.clip(d["odds"] + x @ beta, -36, 36)))
        h = x.T @ (x * (probability * (1 - probability))[:, None]) + np.eye(n_features) * penalty
        update = np.linalg.solve(h, x.T @ (d["y"] - probability) - penalty * beta)
        beta += update
        if np.max(abs(update)) < 1e-7:
            break
    return np.rint(beta * 1024).clip(-512, 512).astype("<i2")


def price(d, beta):
    ev = d["ev"]
    orig = ev["final"]
    use = d["use"]
    p = orig.copy()
    step = np.clip(d["x"][:, : len(beta)] @ (beta.astype(float) / 1024), -1.0, 1.0)
    revised = 1 / (1 + np.exp(-(d["odds"] + step)))
    p[use] = np.rint(revised * TOTAL).clip(1, TOTAL - 1).astype("<u4")
    favored = d["odds"] >= 0
    wrong = favored != d["y"]
    base = orig[use].astype(float) / TOTAL
    q = p[use].astype(float) / TOTAL
    delta = np.where(d["y"] > 0, np.log2(base / q), np.log2((1 - base) / (1 - q)))
    return dict(
        arithmetic_bytes=arithmetic_size(p, ev["bit"]),
        originally_wrong=int(wrong.sum()),
        wrong_bit_delta_bits=float(delta[wrong].sum()),
        correct_bit_delta_bits=float(delta[~wrong].sum()),
    )


def run(a):
    if digest(a.canonical) != CANONICAL_SHA:
        raise ValueError("wrong canonical stream")
    a.out.mkdir(parents=True, exist_ok=False)
    read = partial(load, a.traces, a.canonical, a.extractor.resolve(), a.out)
    train, selection = [read(s) for s in STARTS[:2]]
    # Both challengers use the identical event mask, base odds and support count.
    # Only the spelling model learns which next bit those earlier words favored.
    frozen = {}
    for kind, n in [("ablation", 3), ("spelling", 4)]:
        candidates = []
        for penalty in PENALTIES:
            beta = fit(train, n, penalty)
            score = price(selection, beta)["arithmetic_bytes"]
            candidates.append((score, -penalty, beta))
        score, neg_penalty, beta = min(candidates, key=lambda t: t[:2])
        frozen[kind] = dict(
            coefficients=beta.tolist(),
            penalty=-neg_penalty,
            stored_bytes_per_copy=beta.nbytes,
            selection_bytes=score,
        )
    lock = a.out / "frozen.json"
    lock.write_text(json.dumps(frozen, indent=2) + "\n")
    locked = json.loads(lock.read_bytes())
    spans = []
    for start in STARTS[2:]:
        d = read(start)
        reference = arithmetic_size(d["ev"]["final"], d["ev"]["bit"])
        scored = {}
        for kind, model in locked.items():
            r = price(d, np.array(model["coefficients"], dtype="<i2"))
            r["delta_bytes"] = r["arithmetic_bytes"] - reference
            scored[kind] = r
        spans.append(
            dict(
                start=start,
                events=d["events"],
                trace_sha256=d["source_sha256"],
                targeted_events=d["target_events"],
                eligible_events=len(d["use"]),
                control_bytes=reference,
                models=scored,
            )
        )
    result = dict(
        frozen=locked,
        frozen_sha256=digest(lock),
        fit_start=STARTS[0],
        selection_start=STARTS[1],
        selection_control_bytes=arithmetic_size(selection["ev"]["final"], selection["ev"]["bit"]),
        test_input_bytes=sum(
            json.loads((a.traces / "verified" / f"{s}.json").read_bytes())["end"] - s
            for s in STARTS[2:]
        ),
        test_control_bytes=sum(s["control_bytes"] for s in spans),
        test_delta_bytes={k: sum(s["models"][k]["delta_bytes"] for s in spans) for k in locked},
        eligible_events=sum(s["eligible_events"] for s in spans),
        spans=spans,
    )
    (a.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "spans"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("traces", "canonical", "extractor", "out"):
        p.add_argument(field, type=Path)
    run(p.parse_args())
