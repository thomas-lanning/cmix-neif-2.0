"""Run a bounded sweep of exported candidates, recording successes and failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from evaluate import BASELINE, ROOT, measure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path)
    p.add_argument("suite", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5])
    args = p.parse_args()
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=True)
    generator = Path(__file__).with_name("candidate.py")
    failures = 0
    for fraction in args.fractions:
        with tempfile.TemporaryDirectory(dir=root) as temp:
            tmp = Path(temp)
            shutil.copy(generator, tmp / "hypothesis.py")
            model = tmp / "candidate.c5"
            try:
                # Use a fresh process so failed exports cannot leak model state into the next trial.
                cmd = [sys.executable, str(tmp / "hypothesis.py"), str(BASELINE), str(model), str(fraction)]
                subprocess.run(cmd, check=True, timeout=3600, env={**os.environ, "NEIF_SOURCE_ROOT": str(ROOT)})
                r = measure(args.replay.resolve(), args.suite.resolve(), model, root, f"row fraction {fraction:g}")
                h = r["candidate_sha256"]
                if r["search_proxy_bytes"] < 0 and r["confirmation_proxy_bytes"] < 0:
                    model.replace(root / f"{h}.c5")
                r = {k:r[k] for k in ("candidate_sha256","search_proxy_bytes","confirmation_proxy_bytes","twice_weight_delta")}
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                failures += 1
                r = {"error":str(exc)}
            source = (tmp / "hypothesis.py").read_bytes()
            code_hash = hashlib.sha256(source).hexdigest()
            (root / f"hypothesis-{code_hash}.py").write_bytes(source)
            with (root / "sweep.jsonl").open("a") as log:
                print(json.dumps(dict(fraction=fraction, hypothesis_sha256=code_hash, **r)), file=log)
            print(r, flush=True)
    if failures:
        raise SystemExit(f"{failures} failed trial(s); see sweep.jsonl")


if __name__ == "__main__":
    main()
