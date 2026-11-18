"""Compare frozen final-bit calibrators on an independently captured native batch."""
import argparse
import json
from pathlib import Path

import numpy as np

from entropy_spans import run
from fitted_spans import coder, predict, trace


def measure(exe, data, model, blocks):
    settings = json.loads(model.read_bytes())
    frozen = settings["static"]["frozen"]
    table = np.array(frozen["table"], dtype="i1")
    online = tuple(settings["online"]["frozen_parameters"])
    hashes = dict(reversed(line.split()) for line in (data / "streams.sha256").read_text().splitlines())
    report = {}
    for block in blocks:
        path = data / f"events-{block}.bin"
        meta, events = trace(path, hashes[path.name])
        row = run(exe.resolve(), path, data / "oracle-prefix.bin", online)[online]
        if coder(*meta[:2], events["p"], events["bit"]) != row["base_bytes"]:
            raise ValueError("fitted-table arithmetic control differs")
        adjusted = predict(events, frozen["scope"], table)
        changed = int(coder(*meta[:2], adjusted, events["bit"]))
        report[block] = dict(events=len(events), reference_bytes=row["base_bytes"],
                             online_delta_bytes=row["delta_bytes"],
                             fitted_delta_bytes=changed - row["base_bytes"])
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("executable", "data", "model"):
        p.add_argument(field, type=Path)
    p.add_argument("blocks", type=int, nargs="+")
    args = p.parse_args()
    print(json.dumps(measure(args.executable, args.data, args.model, args.blocks), indent=2))
