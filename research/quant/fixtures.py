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
    parser.add_argument("--midpoints", action="store_true", help="sample the midpoint of each equal-width stream interval")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--start", type=int, help="first allowed source position in the dump (includes its 6-byte header)")
    parser.add_argument("--stop", type=int, help="exclusive source position in the dump")
    parser.add_argument("--compact-priors", type=Path, help="selected binary prior rows")
    parser.add_argument("--compact-plan", type=Path, help="row ranges represented in the compact priors")
    parser.add_argument("--plan-only", action="store_true", help="write selected C40 prior ranges to the output JSON")
    args = parser.parse_args()
    dump = args.scratch / "dump"
    priors = args.scratch / "priors"
    records, header, _ = dumpv2.lade(dump / "paket-zeilen.bin")
    if not header:
        raise ValueError("v2 dump header required")
    origins = np.memmap(priors / "ppmd-quelle.u8", dtype="u1", mode="r")
    raw_tokens = np.memmap(priors / "ppmd-bytes.uint8", dtype="u1", mode="r")
    if args.plan_only:
        if args.selection != "40000000" or args.compact_plan or args.compact_priors:
            raise ValueError("C40 prior planning uses the full token and origin files")
        raw_prior = None
    elif args.compact_priors:
        from compact_priors import CompactRaster, PLAN
        raw_prior = CompactRaster(args.compact_priors, args.compact_plan or PLAN)
    else:
        if args.compact_plan:
            raise ValueError("a compact plan requires the corresponding binary prior rows")
        raw_prior = np.memmap(priors / "ppmd-probs.float16", dtype="<u2", mode="r").reshape(-1, 208)
    stream = np.flatnonzero(origins == 0)
    expected = 37_565_854 if args.selection == "40000000" else 501_062_287
    if (len(stream) != expected or len(origins) != len(raw_tokens) or
            (raw_prior is not None and len(raw_prior) != len(origins))):
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
    if not args.plan_only:
        args.output.mkdir()
    spans = []
    windows = []
    used = set()
    for exclusion in args.exclude_manifest:
        other = json.loads(exclusion.read_bytes())
        if other["selection"] != args.selection:
            raise ValueError("excluded document IDs must refer to the same raw dump")
        used.update(w["document"] for w in other["windows"])
    first = 45_000_000 if args.selection == "full" else 15_282_108
    first = args.start if args.start is not None else first
    stop = args.stop if args.stop is not None else expected
    if not 0 <= first < stop-200_000 or stop > expected:
        raise ValueError("invalid sampling interval")
    offsets = np.linspace(first, stop - 200_000, args.windows, dtype=np.int64)
    if args.midpoints:
        edges = np.linspace(first, stop - 200_000, args.windows+1, dtype=np.int64)
        offsets = (edges[1:] + edges[:-1]) // 2
    for offset in offsets:
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
            if args.stop is not None and np.any(target[:end][scored] >= stop):
                continue
            if not np.array_equal(target_token[:end][scored], tokens[target[:end][scored]]):
                raise ValueError("target tokens disagree with the prior run")
            valid = indices[:end] > 0
            selected = stream[indices[:end][valid] - 1]
            if args.plan_only:
                spans.append((int(selected.min()), int(selected.max())+1))
                windows.append(dict(document=int(doc)))
                used.add(doc)
                break
            rows = np.zeros(end, dtype=ROW)
            for name in ("tok", "p1", "p2", "mb", "mc"):
                rows[name] = fields[name][:end]
            rows["reset"][0] = 1
            rows["target"] = np.where(scored, target_token[:end], 0)
            rows["score"] = scored
            rows["prior"] = fallback
            # Upstream trims injected rows, then looks up prior[target - 1].
            rows["prior"][valid] = raw_prior[selected]
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
    if args.plan_only:
        from compact_priors import PLAN
        reference = json.loads(PLAN.read_bytes())
        if len(origins) != reference["prior_rows"]:
            raise ValueError("unexpected number of C40 rows")
        merged = []
        for lo, hi in sorted(spans):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        with args.output.open("x") as f:
            json.dump(dict(selection=args.selection, windows=windows, prior_rows=len(origins),
                           raw_sha256=reference["raw_sha256"], spans=merged), f)
        return
    manifest = args.output / "manifest.tmp"
    meta=dict(format=1, selection=args.selection,
        input_sha256="159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc",
        windows=windows)
    if args.midpoints:
        meta["sampling"] = "midpoints of equal-width stream intervals"
    if args.start is not None or args.stop is not None:
        meta["source_interval"] = [first, stop]
        # Count the target opportunities in the same source interval as the
        # expensive mixed-coder run. Excludes each document's closing target.
        count = 0
        low = max(0, int(np.searchsorted(src_at_start, first))-1)
        high = min(len(starts)-1,int(np.searchsorted(src_at_start, stop))+1)
        for doc in range(low, high):
            lo, hi = starts[doc:doc+2]
            f = dumpv2.felder(records[lo:hi])
            target, _, score, _ = dumpv2.zielspalten(f, np.array([0,hi-lo]),208)
            count += int(np.count_nonzero(score & (target>=first) & (target<stop)))
        meta["interval_nonterminal_targets"] = count
    manifest.write_text(json.dumps(meta, indent=2) + "\n")
    manifest.replace(args.output / "manifest.json")


if __name__ == "__main__":
    main()
