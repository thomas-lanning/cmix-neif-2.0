"""Hash the complete native C40 prior stream and retain the twelve benchmark spans."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


ROW_BYTES = 208*2
PLAN = Path(__file__).with_name("c40-ranges.json")


def collect(source, destination, plan):
    spans = [(lo*ROW_BYTES,hi*ROW_BYTES) for lo,hi in plan["spans"]]
    hasher = hashlib.sha256()
    offset = 0
    with destination.open("xb") as out:
        while data := source.read(8 << 20):
            hasher.update(data)
            end = offset + len(data)
            for lo, hi in spans:
                a,b = max(offset,lo),min(end,hi)
                if a < b:
                    out.write(memoryview(data)[a-offset:b-offset])
            offset = end
    if offset != plan["prior_rows"]*ROW_BYTES or hasher.hexdigest() != plan["raw_sha256"]:
        raise ValueError("the complete generated PPMd stream differs from the independent reference")
    if destination.stat().st_size != sum(hi-lo for lo,hi in spans):
        raise ValueError("incomplete selected prior rows")


class CompactRaster:
    def __init__(self, path):
        plan = json.loads(PLAN.read_bytes())
        spans = np.array(plan["spans"],dtype=np.int64)
        self.starts,self.stops = spans.T
        self.offsets = np.cumsum(np.r_[0,(self.stops-self.starts)[:-1]])
        self.total = plan["prior_rows"]
        self.rows = np.memmap(path,dtype="<u2",mode="r").reshape(-1,208)
        if len(self.rows) != (self.stops-self.starts).sum():
            raise ValueError("invalid compact raster length")

    def __len__(self):
        return self.total

    def __getitem__(self, ids):
        span = np.searchsorted(self.starts,ids,side="right")-1
        if np.any(span<0) or np.any(ids>=self.stops[span]):
            raise ValueError("requested prior row falls outside the compact benchmark")
        return self.rows[self.offsets[span]+ids-self.starts[span]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    a = parser.parse_args()
    if not a.command or a.destination.exists():
        raise SystemExit("a native prior producer and a fresh destination are required")
    temporary = a.destination.with_suffix(".partial")
    read_fd, write_fd = os.pipe()
    env = dict(os.environ,NEIF_TF1_SAVE_PRIORS=f"/proc/self/fd/{write_fd}")
    with os.fdopen(read_fd,"rb") as source, os.fdopen(write_fd,"wb") as sink:
        with subprocess.Popen(a.command,env=env,pass_fds=(write_fd,)) as producer:
            sink.close()
            try:
                collect(source,temporary,json.loads(PLAN.read_bytes()))
                if producer.wait():
                    raise ValueError("native PPMd producer failed")
            finally:
                if producer.poll() is None:
                    producer.kill()
    temporary.replace(a.destination)
