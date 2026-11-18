"""Measure how clustered semantic-cache matches should vote on the next byte."""
import argparse
import json
from pathlib import Path

import numpy as np

from cache_replay import MIX, load

RING = 1 << 24
RADII = (8, 64, 512, 4096)
POWERS = (-1., -.5, -.25, 0., .25, .5, 1.)


def report(rows, radius, power):
    losses = []
    for row in rows:
        ordered = row["native"]
        ids = row["ring_ids"][ordered].astype(np.int64)
        distance = abs(ids[:, None] - ids)
        count = (np.minimum(distance, RING - distance) <= radius).sum(axis=1)
        scores = row["native_scores"]
        weights = np.exp((scores - scores[0]) / .020)
        match = row["values"][ordered] == row["target"]
        prior = (1-MIX) * row["tf"]
        base = prior + MIX * (weights[match].sum() / weights.sum())
        weights *= count ** power
        revised = prior + MIX * (weights[match].sum() / weights.sum())
        losses.append(np.log2(np.maximum(base, 1.e-30) / np.maximum(revised, 1.e-30)))
    losses = np.asarray(losses)
    return dict(queries=len(rows), bits=losses.sum(axis=0).tolist(),
                quarter_bits=[float(s.sum()) for s in np.array_split(losses[:, 1], 4)])


def evaluate(root, starts, output, frozen):
    def read(start):
        rows, _, _ = load(root / f"cache-{start}.bin", root / "results" / f"receipt-{start}.json",
                          root / "output.cmix.temp")
        return rows
    if frozen is None:
        rows = read(starts[0])
        split = 3 * len(rows) // 4
        choices = {}
        for radius in RADII:
            choices[radius] = min(POWERS, key=lambda p: (report(rows[:split], radius, p)["bits"][1],
                                                         abs(p)))
        options = [(8, 0.)] + list(choices.items())
        radius, power = min(options, key=lambda rp: (report(rows[split:], *rp)["bits"][1], abs(rp[1])))
        frozen = dict(radius=radius, power=power, fit_start=starts[0], training_queries=split,
                      selection_queries=len(rows)-split)
        starts = starts[1:]
    output.mkdir(parents=True, exist_ok=True)
    record = output / "frozen.json"
    content = json.dumps(frozen, indent=2) + "\n"
    if record.exists() and record.read_text() != content:
        raise ValueError("existing frozen model differs")
    record.write_text(content)
    results = {s: report(read(s), frozen["radius"], frozen["power"]) for s in starts}
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(dict(frozen=frozen, evaluation=results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("starts", type=int, nargs="+")
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    evaluate(args.root, args.starts, args.output,
             json.loads(args.frozen.read_bytes()) if args.frozen else None)
