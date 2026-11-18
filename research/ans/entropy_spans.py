"""Select on three early blocks, then score a frozen online calibrator on three later blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BLOCKS = (118, 120, 122, 128, 132, 136)
IDENTITY = (0, 1, 1)


def run(exe, path, oracle, choice=()):
    lines = subprocess.check_output([exe, path, oracle, *map(str, choice)], text=True).splitlines()
    scores = {}
    for line in lines:
        family, prior, strength, n, offset, reference, candidate = map(int, line.split())
        scores[family, prior, strength] = dict(events=n, offset=offset, base_bytes=reference,
                                               candidate_bytes=candidate, delta_bytes=candidate-reference)
    return scores


def study(exe, data, output):
    traces = {b: data / f"events-{b}.bin" for b in BLOCKS}
    expected = {name: digest for digest, name in (line.split() for line in
                (data / "streams.sha256").read_text().splitlines())}
    for path in traces.values():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[path.name]:
            raise ValueError(f"trace hash mismatch for {path.name}")
    oracle = data / "oracle-prefix.bin"
    scores = [run(exe, traces[b], oracle) for b in BLOCKS[:2]]
    # Each family nominates one prior and strength using only the first two blocks.
    def fit(choice):
        return (sum(s[choice]["delta_bytes"] for s in scores), choice)
    candidates = [IDENTITY] + [min((c for c in scores[0] if c[0] == family), key=fit)
                              for family in range(1, 6)]
    select = {c: run(exe, traces[BLOCKS[2]], oracle, c)[c] for c in candidates}
    winner = min(candidates, key=lambda c: (select[c]["delta_bytes"], c))
    output.mkdir(exist_ok=True, parents=True)
    record = dict(fit_blocks=BLOCKS[:2], selection_block=BLOCKS[2],
                  nominees=[dict(parameters=c, training_delta_bytes=fit(c)[0],
                                 selection=select[c]) for c in candidates],
                  frozen_parameters=winner)
    frozen = output / "frozen.json"
    encoded = json.dumps(record, indent=2)+"\n"
    if frozen.exists() and frozen.read_text() != encoded:
        raise ValueError("existing freeze record differs")
    frozen.write_text(encoded)
    heldout = {b: run(exe, traces[b], oracle, winner)[winner] for b in BLOCKS[3:]}
    result = dict(**record, heldout=heldout,
                  heldout_delta_bytes=sum(x["delta_bytes"] for x in heldout.values()),
                  oracle_prefix_sha256=hashlib.sha256(oracle.read_bytes()).hexdigest(),
                  unit_canonical_bytes=131072, trace_hashes=expected)
    (output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path); p.add_argument("data", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args();study(a.executable.resolve(), a.data, a.output)
