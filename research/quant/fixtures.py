"""Select disjoint document-aligned windows from the actual cmix coder dumps."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/tf/2_package"))
import dumpv2

ROW = np.dtype([(name, "u1") for name in ("tok", "p1", "p2", "mb", "mc", "target", "score", "reset")]
               + [("prior", "<u2", (208,))])
MAGIC = b"NEIFQ001"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("selection", choices=["40000000", "full"])
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--exclude-manifest", type=Path)
    args = parser.parse_args()
    dump = args.scratch / "dump"
    priors = args.scratch / "priors"
    records, header, _ = dumpv2.lade(dump / "paket-zeilen.bin")
    if not header:
        raise ValueError("v2 dump header required")
    origins = np.memmap(priors / "ppmd-quelle.u8", dtype="u1", mode="r")
    raw_tokens = np.memmap(priors / "ppmd-bytes.uint8", dtype="u1", mode="r")
    raw_prior = np.memmap(priors / "ppmd-probs.float16", dtype="<u2", mode="r").reshape(-1, 208)
    stream = np.flatnonzero(origins == 0)
    expected = 37_565_854 if args.selection == "40000000" else 501_062_287
    if len(stream) != expected or len(origins) != len(raw_tokens) or len(raw_prior) != len(origins):
        raise ValueError("incomplete or mismatched rasters")
    tokens = raw_tokens[stream]
    # Check the native prior targets against the original transformed bytes (six-byte segment header).
    bitmap = (ROOT / "training/tf/2_package/vocab-bitmap.u8").read_bytes()
    lut = np.full(256, 255, dtype="u1")
    vocab = [i for i in range(256) if bitmap[i//8] & (1 << (i%8))]
    lut[vocab] = np.arange(len(vocab))
    canonical = np.memmap(args.scratch / "bed/censpruef_strom.bin", dtype="u1", mode="r")
    if not np.array_equal(tokens[6:], lut[canonical[:expected-6]]):
        raise ValueError("byte-to-prior alignment differs from the canonical coder stream")
    starts = np.flatnonzero(records["fl"] & dumpv2.DOKANFANG)
    if args.windows < 12 or args.windows > 96 or args.windows % 2:
        raise ValueError("choose an even number of windows from 12 through 96")
    if len(starts) < args.windows:
        raise ValueError("insufficient document starts")
    # Stream byte 15,282,108 is the end of the title gate after the three setup articles.
    # Aim at byte quantiles, then choose the first full document with a real body.
    src_at_start = np.maximum.accumulate(np.maximum(records["src"], 0))[starts]
    text = (ROOT / "src/tf1/tf1_prior0_set_v14.h").read_text().split("#define PRIOR0_INIT", 1)[1]
    fallback = np.array([int(x, 16) for x in re.findall(r"0x[0-9a-f]+", text)], dtype="<u2")
    if hashlib.sha256(fallback.tobytes()).hexdigest() != "ceb297e328d9af5daa72f47b750267b1c1978e8d382d1d254692fe2dc31cbafc":
        raise ValueError("unexpected prior0")
    args.output.mkdir()
    windows = []
    used = set()
    if args.exclude_manifest:
        other = json.loads(args.exclude_manifest.read_bytes())
        if other["selection"] != args.selection:
            raise ValueError("excluded document IDs must refer to the same raw dump")
        used.update(w["document"] for w in other["windows"])
    first = 45_000_000 if args.selection == "full" else 15_282_108
    for offset in np.linspace(first, expected - 200_000, args.windows, dtype=np.int64):
        for doc in range(int(np.searchsorted(src_at_start, offset)), len(starts) - 1):
            if doc in used:
                continue
            lo, hi = starts[doc:doc+2]
            if hi-lo < 520 or hi-lo > 131072:
                continue
            fields = dumpv2.felder(records[lo:hi])
            # Omit the terminal prediction, whose target is never fed in the dump.
            target, target_token, scored, _ = dumpv2.zielspalten(fields, np.array([0, hi-lo]), 208)
            indices = dumpv2.priorziel(fields, target)
            # Replay from a true document reset. The first 256 fed rows warm the model.
            end = min(hi-lo-1, 4352)
            scored = scored[:end].copy()
            scored[:256] = False
            if scored.sum() < 256:
                continue
            if not np.array_equal(target_token[:end][scored], tokens[target[:end][scored]]):
                raise ValueError("target tokens disagree with the prior run")
            rows = np.zeros(end, dtype=ROW)
            for name in ("tok", "p1", "p2", "mb", "mc"):
                rows[name] = fields[name][:end]
            rows["reset"][0] = 1
            rows["target"] = np.where(scored, target_token[:end], 0)
            rows["score"] = scored
            rows["prior"] = fallback
            valid = indices[:end] > 0
            # Upstream trims injected rows, then looks up prior[target - 1].
            rows["prior"][valid] = raw_prior[stream[indices[:end][valid] - 1]]
            data = MAGIC + rows.tobytes()
            file = f"window-{len(windows):02d}.bin"
            (args.output / file).write_bytes(data)
            windows.append(dict(file=file, sha256=hashlib.sha256(data).hexdigest(),
                                document=int(doc), stream_byte=int(src_at_start[doc]),
                                rows=int(end), targets=int(scored.sum())))
            used.add(doc)
            break
        else:
            raise ValueError(f"no usable document near {offset}")
    manifest = args.output / "manifest.tmp"
    manifest.write_text(json.dumps(dict(format=1, selection=args.selection,
        input_sha256="159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc",
        windows=windows), indent=2) + "\n")
    manifest.replace(args.output / "manifest.json")


if __name__ == "__main__":
    main()
