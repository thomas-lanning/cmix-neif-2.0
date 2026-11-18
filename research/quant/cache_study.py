"""Drive the release-profiled decoder and verify each cache trace against independent input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from cache_replay import digest

# Digests from the independent encoder preprocessing of the official enwik9.
CHECKS={
    200000:"a3781682da3dc803ae1a7ee93630b8158ad1e8a76f58af26095a5c1050d1dd9e",
    18000000:"ab41f3c44d66fc49d7d9eba31c5ee3e2794fc99e2d3b6fe9d1f431af16f977a7",
    30000000:"a902dbfe09e299490b83c397228772e2f5b773428c2bdfafd9a551543f1edaba",
    40000000:"2e553f20043278bd04a18d236412541c054525dfaca8836a160ffd065539f481",
}


def run(executable,payload,results,deadline):
    results.mkdir()
    canonical=Path("output.cmix.temp")
    model=results/"frozen.json"
    with Path("native.log").open("wb") as log:
        worker=subprocess.Popen([str(executable.resolve()),"-d",str(payload.resolve()),"output"],
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for stop,expected in CHECKS.items():
                while not canonical.is_file() or canonical.stat().st_size<stop:
                    if time.monotonic()>=deadline:
                        print("Worker time budget reached before",stop,flush=True);return
                    if worker.poll() is not None:raise RuntimeError(f"native decoder exited before {stop}")
                    time.sleep(5)
                with canonical.open("rb") as f:actual=hashlib.sha256(f.read(stop)).hexdigest()
                if actual!=expected:raise ValueError(f"independent canonical reference differs at {stop}")
                print("Verified canonical bytes",stop,actual,flush=True)
                if stop==200000:continue
                start=stop-2000000
                marker=Path(f"cache-{start}.done")
                # The stdio buffer may flush an exact multiple before the next
                # SemByteStart writes the completion marker.
                while not marker.is_file():
                    if worker.poll() is not None or time.monotonic()>deadline:raise ValueError("missing capture marker")
                    time.sleep(1)
                eligible,sampled=map(int,marker.read_text().split())
                trace=Path(f"cache-{start}.bin")
                receipt=results/f"receipt-{start}.json"
                info=dict(start=start,stop=stop,eligible_queries=eligible,sampled_queries=sampled,
                          canonical_prefix_sha256=actual,canonical_prefix_bytes=stop,
                          trace_sha256=digest(trace) if sampled else None,trace_bytes=trace.stat().st_size if sampled else 0)
                receipt.write_text(json.dumps(info,indent=2)+"\n")
                if sampled<48:
                    print("Insufficient eligible queries at",start,sampled,flush=True);continue
                action="test" if model.exists() else "train"
                out=results/f"result-{start}.json"
                subprocess.run([sys.executable,Path(__file__).with_name("cache_replay.py"),action,
                                trace,receipt,canonical,model,out],check=True)
                r=json.loads(out.read_bytes())
                print("Cache study",start,action,sampled,r["frozen_selection"],
                      r["metrics"][r["frozen_selection"]]["delta_bits_by_mixture"],flush=True)
        finally:
            if worker.poll() is None:os.killpg(worker.pid,signal.SIGTERM)
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGKILL);worker.wait()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("executable","payload","results"):parser.add_argument(n,type=Path)
    parser.add_argument("--minutes",type=float,default=310)
    a=parser.parse_args()
    run(a.executable,a.payload,a.results,time.monotonic()+a.minutes*60)
