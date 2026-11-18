"""Compare a bounded decode by two complete self-extracting release archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def prefix(archive,folder,size,timeout):
    folder.mkdir()
    exe=folder/"archive9"
    shutil.copyfile(archive,exe)
    exe.chmod(0o755)
    end=time.monotonic()+timeout
    with (folder/"decoder.log").open("wb") as log:
        process=subprocess.Popen([str(exe.resolve())],cwd=folder,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            output=folder/".cmix.temp"
            while time.monotonic()<end:
                if output.exists() and output.stat().st_size>=size:
                    with output.open("rb") as f:return f.read(size)
                if process.poll() is not None:
                    raise RuntimeError(f"decoder exited {process.returncode} before producing the test prefix")
                time.sleep(2)
            raise TimeoutError("release decoder exceeded the prefix-test budget")
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("published","candidate","destination"):p.add_argument(name,type=Path)
    p.add_argument("--bytes",type=int,default=200000)
    p.add_argument("--timeout",type=int,default=900)
    a=p.parse_args();a.destination.mkdir()
    if a.bytes<1 or a.timeout<1:raise ValueError("positive sample length and deadline required")
    original=prefix(a.published,a.destination/"reference",a.bytes,a.timeout)
    trial=prefix(a.candidate,a.destination/"candidate",a.bytes,a.timeout)
    if len(original)!=a.bytes or original!=trial:raise ValueError("release decoded bytes differ")
    result=dict(bytes=len(original),sha256=hashlib.sha256(original).hexdigest(),
                published_archive_sha256=hashlib.sha256(a.published.read_bytes()).hexdigest(),
                reassembled_archive_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
                self_extracting_prefix_equal=True)
    (a.destination/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
