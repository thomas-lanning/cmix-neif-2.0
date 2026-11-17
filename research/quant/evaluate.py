"""Compare one trusted same-geometry NP2 export with the published model."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5"
REFERENCE_SHA = "729694cba4969b7d833686ed3f597ff04d3b226da849d5f8004ec020a0ed9563"
FULL_TARGETS = 487_298_491


def digest(data):
    return hashlib.sha256(data).hexdigest()


def model_id(path):
    data = path.read_bytes()
    if not data.startswith(b"FX2TFWP5"):
        raise ValueError("expected a compatible NP2 export from the fixed-geometry model")
    fnv = 14695981039346656037
    for byte in data:
        fnv = ((fnv ^ byte) * 1099511628211) & 0xffffffffffffffff
    return digest(data), format(fnv, "016x"), len(data)


def replay(executable, path, model, windows, suite):
    result = []
    for w in windows:
        command = [str(executable), str(path), model[1], str(suite / w["file"])]
        got = json.loads(subprocess.check_output(command, text=True, timeout=300))
        if (got["rows"], got["targets"]) != (w["rows"], w["targets"]):
            raise ValueError("replay counts disagree with the pinned fixture")
        result.append(got)
    return result


def measure(exe, suite, candidate, store, label):
    store.mkdir(exist_ok=True, parents=True)
    manifest = (suite / "manifest.json").read_bytes()
    info = json.loads(manifest)
    windows = info["windows"]
    if (not 12 <= len(windows) <= 96 or len(windows) % 2 or
            len({w["file"] for w in windows}) != len(windows) or info["format"] != 1):
        raise ValueError("expected an even number of distinct windows in suite version 1")
    for w in windows:
        path = suite / w["file"]
        if path.parent != suite or digest(path.read_bytes()) != w["sha256"]:
            raise ValueError(f"unexpected fixture {path}")
    base = model_id(BASELINE)
    if base[0] != REFERENCE_SHA:
        raise ValueError("published model hash differs")
    model = model_id(candidate)
    # Cache only immutable measurements made by this evaluator and native binary.
    key = digest(manifest + exe.read_bytes() + Path(__file__).read_bytes() + bytes.fromhex(base[0]))
    ref_file = store / f"reference-{key}.json"
    with (store / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not ref_file.exists():
            ref = replay(exe, BASELINE, base, windows, suite)
            with tempfile.NamedTemporaryFile(dir=store, mode="w", delete=False) as out:
                json.dump(ref, out)
            Path(out.name).replace(ref_file)
        else:
            ref = json.loads(ref_file.read_text())
    started = time.monotonic()
    trial = replay(exe, candidate, model, windows, suite)
    replay_seconds = time.monotonic() - started
    window_deltas = [(t["bits"]-r["bits"])/r["targets"] for r,t in zip(ref,trial)]
    weight_delta = 2 * (model[2] - base[2])
    # Each window represents one evenly spaced stream position. Report the
    # equal-window mean and its constituent deltas so coverage is inspectable.
    def proxy(group):
        return weight_delta + statistics.mean(group)*FULL_TARGETS/8
    result = dict(label=label, candidate_sha256=model[0], candidate_bytes=model[2],
                  reference_sha256=base[0], suite_sha256=digest(manifest), evaluator=key,
                  selection=info["selection"], twice_weight_delta=weight_delta, replay_seconds=replay_seconds,
                  bits_per_target_delta_by_window=window_deltas, reference=ref, candidate=trial,
                  search_proxy_bytes=proxy(window_deltas[::2]),
                  confirmation_proxy_bytes=proxy(window_deltas[1::2]),
                  proxy_delta_bytes=proxy(window_deltas), time_unix=time.time())
    with (store / "trials.jsonl").open("a") as journal:
        fcntl.flock(journal, fcntl.LOCK_EX)
        print(json.dumps(result, separators=(",", ":")), file=journal)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path)
    p.add_argument("suite", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("--label", default="manual")
    args = p.parse_args()
    r = measure(args.replay.resolve(), args.suite.resolve(), args.candidate.resolve(), args.results, args.label)
    for k in ("twice_weight_delta","search_proxy_bytes","confirmation_proxy_bytes","proxy_delta_bytes","replay_seconds"):
        print(f"METRIC {k}={r[k]:.6f}")


if __name__ == "__main__":
    main()
