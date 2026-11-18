"""Losslessly repackage all published tensors with a static multi-symbol rANS coder."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/tf/3_train/export/np2"))
import np2_container as np2

MAGIC = b"NEIFANS1"
PRECISION = 12
TOTAL = 1 << PRECISION
LOW = 1 << 23
DTYPES = ("i1","<u2","<f4","<i4")


def frequencies(symbols):
    counts = np.bincount(symbols, minlength=15)
    norm = counts * (TOTAL - np.count_nonzero(counts)) / len(symbols)
    f = np.floor(norm).astype(np.int32) + (counts != 0)
    remaining = TOTAL - f.sum()
    order = np.argsort(norm - np.floor(norm), kind="stable")[::-1]
    f[order[:remaining]] += 1
    if f.sum() != TOTAL or np.any((counts != 0) & (f == 0)):
        raise ValueError("invalid normalization")
    return f.tolist()


def encode(symbols, freq):
    start = np.cumsum([0]+freq[:-1]).tolist()
    thresh = [((LOW >> PRECISION) << 8)*f for f in freq]
    state, data = LOW, bytearray()
    for s in reversed(symbols.tolist()):
        f = freq[s]
        while state >= thresh[s]:
            data.append(state & 255)
            state >>= 8
        state = ((state//f) << PRECISION) + state%f + start[s]
    return struct.pack("<I",state) + data[::-1]


def decode(data, n, freq):
    if len(data)<4 or len(freq)!=15 or sum(freq)!=TOTAL or any(f<0 for f in freq):
        raise ValueError("invalid weight rANS frame")
    sym = [s for s,f in enumerate(freq) for _ in range(f)]
    start = np.cumsum([0]+freq[:-1]).tolist()
    state, = struct.unpack_from("<I",data)
    pos, raw = 4, bytearray(n)
    for i in range(n):
        slot = state & (TOTAL-1)
        s = sym[slot]
        raw[i] = s
        state = freq[s]*(state >> PRECISION) + slot-start[s]
        while state < LOW:
            if pos >= len(data): raise ValueError("truncated rANS stream")
            state = (state << 8) | data[pos]
            pos += 1
    if state != LOW or pos != len(data):
        raise ValueError("weight rANS terminal mismatch")
    return np.frombuffer(raw,dtype="i1") - 7


def pack(tensors):
    descriptors, weights, other = [], bytearray(), bytearray()
    for t in tensors:
        a = t.a
        if t.name in ("rope.sin","rope.cos"):
            mode, extra = "rope", None
        elif t.code == 0 and a.size and a.min() >= -7 and a.max() <= 7:
            ids=(a.ravel()+7).astype("u1")
            freq=frequencies(ids)
            data=encode(ids,freq)
            weights.extend(data)
            mode,extra="ans",[len(data),freq]
        else:
            mode="bytes"
            b=a
            if t.code==2 and np.all((a.view("<u4") & 65535)==0):
                mode="high16"
                b=(a.view("<u4") >> 16).astype("<u2")
            # Group the byte planes to retain exponent and scale regularities.
            other.extend(b.reshape(-1).view("u1").reshape(-1,b.itemsize).T.tobytes())
            extra=None
        descriptors.append([t.name,t.code,a.shape,mode,extra])
    header=zlib.compress(json.dumps(descriptors,separators=(",",":")).encode(),9)
    side=zlib.compress(other,9)
    return MAGIC+struct.pack("<II",len(header),len(side))+header+side+weights


def unpack(data):
    if data[:8]!=MAGIC:
        raise ValueError("unknown weight format")
    n,m=struct.unpack_from("<II",data,8)
    descriptors=json.loads(zlib.decompress(data[16:16+n]))
    side=memoryview(zlib.decompress(data[16+n:16+n+m]))
    wpos,spos=16+n+m,0
    tensors=[]
    inv=None
    for name,code,shape,mode,extra in descriptors:
        count=int(np.prod(shape))
        if mode=="rope":
            if inv is None: raise ValueError("missing rotary frequencies")
            a=np2.wc.cuda_rope_table(inv,shape[0],cos=name=="rope.cos")
        elif mode=="ans":
            size,freq=extra
            a=decode(data[wpos:wpos+size],count,freq)
            wpos+=size
        else:
            dt=np.dtype("<u2" if mode=="high16" else DTYPES[code])
            size=dt.itemsize*count
            planes=np.frombuffer(side[spos:spos+size],dtype="u1").reshape(dt.itemsize,count)
            a=np.ascontiguousarray(planes.T).reshape(-1).view(dt)
            spos+=size
            if mode=="high16":
                a=(a.astype("<u4") << 16).view("<f4")
        t=np2.Ent(name,code,a.reshape(shape))
        if name=="rope.inv_freq": inv=t.a.reshape(-1)
        tensors.append(t)
    if spos!=len(side) or wpos!=len(data):
        raise ValueError("unused weight container data")
    return tensors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    for op in ("repack","unpack"):
        sp=sub.add_parser(op)
        sp.add_argument("input",type=Path)
        sp.add_argument("output",type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise SystemExit("choose a fresh output path")
    if a.action=="unpack":
        np2.write_raw(unpack(a.input.read_bytes()),a.output)
        return
    original=np2.read_container(str(a.input))
    encoded=pack(original)
    readback=unpack(encoded)
    if len(original)!=len(readback) or not all(t.same(u) for t,u in zip(original,readback)):
        raise ValueError("decoded tensors differ from published NP2")
    with a.output.open("xb") as f: f.write(encoded)
    print(json.dumps(dict(original_bytes=a.input.stat().st_size, rans_bytes=len(encoded),
                          tensors=len(original), exact=True, sha256=hashlib.sha256(encoded).hexdigest())))


if __name__=="__main__":
    main()
