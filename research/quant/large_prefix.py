"""Measure a frozen model on a matched 40M prefix and verify it with a fresh decoder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from cache_windows import DIGESTS
from evaluate import BASELINE, REFERENCE_SHA, ROOT, model_id
from transfer import checkpoints

SHORT = 20_000_000
LONG = 40_000_000
CANONICAL_SHA = "c35d3073eeba70f22a2e08a50202b53eb95466e3e9f1f2702f6f9b3f96fd8718"
FITTED_STABLE_SHA = "3c7e8f44e9a8f79565331ece6a4bd4b1b80a1d872cc7fc98ae12c8420fcd22d6"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def invoke(operation, source, destination, fnv, canonical=None):
    env = dict(os.environ, NEIF_CANDIDATE_FNV=fnv)
    if canonical:
        env["NEIF_CANONICAL_STREAM"] = str(canonical)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    shutil.copyfile(ROOT / "dictionary/english.dic", ".dict")
    with Path("codec.log").open("wb") as log:
        subprocess.run(["/usr/bin/time", "-v", "./cmix", operation, source, destination],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def encode(args):
    model = BASELINE
    stable = None
    expected_prefix_hash = FITTED_STABLE_SHA
    if args.variant == "weights":
        short = json.loads((args.short / "measurement.json").read_bytes())
        candidate = short["candidate"]
        payload = args.short / "candidate/payload"
        if (candidate["model_sha256"] != args.sha or short["end_checkpoint_canonical_bytes"] != SHORT
                or candidate["encoded_sha256"] != digest(payload)):
            raise ValueError("20M measurement is for a different model or payload")
        model = next(p for p in args.exports.rglob("*.c5") if digest(p) == args.sha)
        # The last byte terminates the short arithmetic stream.
        stable = payload.read_bytes()[:-1]
    elif digest(model) != REFERENCE_SHA:
        raise ValueError("unknown upstream weights")
    elif args.variant == "frozen-layer":
        from allocation import export, np2
        receipt = json.loads(Path(__file__).with_name("frozen-layer-20m.json").read_bytes())
        model = Path("layer-export.c5")
        export(np2.read_container(str(BASELINE)), receipt["specs"], model)
        if digest(model) != receipt["model_sha256"] or receipt["short_input_bytes"] != SHORT:
            raise ValueError("layer export differs from the frozen 20M candidate")
        expected_prefix_hash = receipt["short_stable_sha256"]
    sha, fnv, weight_bytes = model_id(model)
    shutil.copyfile(model, BASELINE.name)
    if digest(args.canonical) != CANONICAL_SHA:
        raise ValueError("unexpected canonical input")
    invoke("-e", args.canonical, "payload", fnv, args.canonical)
    sizes = checkpoints(Path.cwd())
    payload = Path("payload").read_bytes()
    # The research stream contains a five-byte header and one final flush byte.
    if sizes.keys() != {15_282_108, 16_000_000, 17_000_000, 18_000_000, SHORT, 30_000_000, LONG} or len(payload) != sizes[LONG] + 6:
        raise ValueError("incomplete native encode")
    prefix = payload[:sizes[SHORT] + 5]
    if args.variant == "reference":
        with args.published_payload.open("rb") as reference:
            if payload[:-1] != reference.read(len(payload)-1):
                raise ValueError("40M control differs from the published arithmetic payload")
    elif args.variant in ("fitted-mixer", "frozen-layer"):
        if hashlib.sha256(prefix).hexdigest() != expected_prefix_hash:
            raise ValueError("the frozen candidate failed to reproduce its 20M result")
    elif prefix != stable:
        raise ValueError("the frozen weights failed to reproduce their 20M payload")
    report = dict(variant=args.variant, weight_sha256=sha, weight_bytes=weight_bytes,
                  payload_bytes=len(payload), payload_sha256=digest(Path("payload")),
                  checkpoints=sizes, cpu_binary_sha256=digest(Path("cmix")))
    Path("encode.json").write_text(json.dumps(report, indent=2) + "\n")


def decode(args):
    result = json.loads(args.candidate.read_bytes())
    reference = json.loads(args.reference.read_bytes())
    sha, fnv, weight_bytes = model_id(Path(BASELINE.name))
    if (reference["variant"] != "reference" or reference["weight_sha256"] != REFERENCE_SHA
            or result["weight_sha256"] != sha or result["weight_bytes"] != weight_bytes
            or digest(Path("payload")) != result["payload_sha256"]
            or digest(Path("cmix")) != result["cpu_binary_sha256"]):
        raise ValueError("unexpected reference, weights, binary or arithmetic stream")
    invoke("-d", "payload", "output", fnv)
    restored = Path("output.cmix.temp")
    if restored.stat().st_size != LONG or digest(restored) != DIGESTS[LONG]:
        raise ValueError("fresh decoder failed to restore the independent 40M prefix")
    deltas = {int(k): n-reference["checkpoints"][k] for k,n in result["checkpoints"].items()}
    delta = result["payload_bytes"] - reference["payload_bytes"]
    if deltas[LONG] != delta:
        raise ValueError("checkpoint and flushed sizes disagree")
    outcome = dict(model_sha256=sha, canonical_bytes=LONG, canonical_sha256=DIGESTS[LONG],
                   control_payload_bytes=reference["payload_bytes"], candidate_payload_bytes=result["payload_bytes"],
                   payload_delta_bytes=delta, extra_20m_delta_bytes=deltas[LONG]-deltas[SHORT],
                   checkpoint_deltas=deltas, both_weight_copies_delta_bytes=2*(weight_bytes-reference["weight_bytes"]))
    Path("comparison.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome), flush=True)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            print(f"Verified {LONG:,} preprocessed bytes by fresh decoding. "
                  f"Candidate payload {result['payload_bytes']:,} bytes, control {reference['payload_bytes']:,}. "
                  f"Difference {delta:+,} bytes overall and {outcome['extra_20m_delta_bytes']:+,} on the added 20M. "
                  f"Two stored weight copies differ by {outcome['both_weight_copies_delta_bytes']:+,} bytes.", file=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(required=True)
    enc = subs.add_parser("encode")
    enc.add_argument("variant", choices=["reference", "fitted-mixer", "frozen-layer", "weights"])
    enc.add_argument("canonical", type=Path)
    enc.add_argument("--published-payload", type=Path)
    enc.add_argument("--exports", type=Path)
    enc.add_argument("--short", type=Path)
    enc.add_argument("--sha")
    enc.set_defaults(action=encode)
    dec = subs.add_parser("decode")
    dec.add_argument("candidate", type=Path)
    dec.add_argument("reference", type=Path)
    dec.set_defaults(action=decode)
    args = parser.parse_args()
    args.action(args)
