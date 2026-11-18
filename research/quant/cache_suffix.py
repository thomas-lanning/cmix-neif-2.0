"""Reweight semantic neighbours by matching bytes immediately before their stored positions."""
import argparse
import json
from pathlib import Path

import numpy as np

from cache_replay import MIX, load
from cache_windows import WINDOWS

HISTORY = 1 << 25
MAX_SUFFIX = 8


def observations(folder, start):
    output = folder / "output.cmix.temp"
    rows, _, shift = load(folder / f"cache-{start}.bin", folder / "results" / f"receipt-{start}.json", output)
    corpus = np.memmap(output, dtype="u1", mode="r")
    records = []
    for row in rows:
        selected = row["native"]
        old = row["positions"][selected] + shift
        pos = row["pos"] + shift
        matches = (old >= MAX_SUFFIX) & (pos-old+MAX_SUFFIX <= HISTORY)
        length = np.zeros(len(old), dtype=int)
        for depth in range(1, MAX_SUFFIX+1):
            matches &= corpus[np.maximum(0, old-depth)] == corpus[pos-depth]
            length += matches
        weights = np.exp((row["native_scores"] - row["native_scores"][0]) / .020)
        records.append((length, weights, row["values"][selected] == row["target"], row["tf"]))
    return records


def delta(records, model):
    cap, strength = model
    losses = []
    for length, weights, target, tf in records:
        mass = weights[target].sum()/weights.sum()
        revised = weights * np.exp(strength * np.minimum(length, cap))
        mass2 = revised[target].sum()/revised.sum()
        reference = (1-MIX)*tf + MIX*mass
        candidate = (1-MIX)*tf + MIX*mass2
        losses.append(np.log2(np.maximum(reference, 1.e-30)/np.maximum(candidate, 1.e-30)))
    losses = np.asarray(losses)
    return dict(queries=len(losses), bits=losses.sum(axis=0).tolist(),
                quarter_bits=[float(part.sum()) for part in np.array_split(losses[:, 1], 4)])


def study(folder, schedule):
    starts = [start for start, _ in WINDOWS[schedule]]
    fit = observations(folder, starts[0])
    split = len(fit)*3//4
    choices = [(1, 0.)]
    for cap in (1, 2, 4, 8):
        strength = min((-.4, -.2, 0., .2, .4, .8, 1.2),
                       key=lambda x: (delta(fit[:split], (cap, x))["bits"][1], abs(x)))
        choices.append((cap, strength))
    model = min(choices, key=lambda m: (delta(fit[split:], m)["bits"][1], abs(m[1]), m[0]))
    frozen = dict(cap=model[0], strength=model[1], selection=delta(fit[split:], model),
                  fit_start=starts[0], selection_start_query=split)
    path = folder / "results" / "suffix-frozen.json"
    content = json.dumps(frozen, indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError("existing frozen suffix model differs")
    path.write_text(content)
    result = {s: delta(observations(folder, s), model) for s in starts[1:]}
    (folder / "results" / "suffix-holdout.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(frozen=frozen, tests=result)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("schedule", choices=WINDOWS)
    a = p.parse_args()
    study(a.folder, a.schedule)
