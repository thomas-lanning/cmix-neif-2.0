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
from cache_windows import DIGESTS, WINDOWS


def run(executable,payload,results,deadline,schedule):
    results.mkdir()
    canonical=Path("output.cmix.temp")
    model=results/"frozen.json"
    completed=[]
    with Path("native.log").open("wb") as log:
        worker=subprocess.Popen([str(executable.resolve()),"-d",str(payload.resolve()),"output"],
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for start,stop in ((0,200000),*WINDOWS[schedule]):
                expected=DIGESTS[stop]
                while not canonical.is_file() or canonical.stat().st_size<stop:
                    if time.monotonic()>=deadline:
                        raise TimeoutError(f"native decoder reached its time budget before {stop}")
                    if worker.poll() is not None:raise RuntimeError(f"native decoder exited before {stop}")
                    time.sleep(5)
                with canonical.open("rb") as f:actual=hashlib.sha256(f.read(stop)).hexdigest()
                if actual!=expected:raise ValueError(f"independent canonical reference differs at {stop}")
                print("Verified canonical bytes",stop,actual,flush=True)
                if stop==200000:continue
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
                completed.append(dict(start=start,stop=stop,role=action,queries=sampled))
                print("Cache study",start,action,sampled,r["frozen_selection"],
                      r["metrics"][r["frozen_selection"]]["delta_bits_by_mixture"],flush=True)
            summary=dict(schedule=schedule,completed=completed,
                         held_out=sum(row["role"]=="test" for row in completed))
            (results/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            if not summary["held_out"]:
                raise ValueError("no independent cache validation span completed")
        finally:
            if worker.poll() is None:os.killpg(worker.pid,signal.SIGTERM)
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGKILL);worker.wait()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("executable","payload","results"):parser.add_argument(n,type=Path)
    parser.add_argument("--minutes",type=float,default=310)
    parser.add_argument("--schedule",choices=WINDOWS,default="full")
    a=parser.parse_args()
    run(a.executable,a.payload,a.results,time.monotonic()+a.minutes*60,a.schedule)
