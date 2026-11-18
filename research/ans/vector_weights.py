"""Pack vector-quantized MLP weights and escaped original vectors with rANS."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"quant"))
from allocation import np2

MAGIC=b"NEIFVQR1"
TOTAL=4096
LOW=1<<23


def frequencies(values,alphabet):
    count=np.bincount(values,minlength=alphabet)
    active=np.flatnonzero(count)
    if not len(active): return []
    ideal=count[active]*(TOTAL-len(active))/len(values)
    base=1+np.floor(ideal).astype(int)
    rest=TOTAL-base.sum()
    order=np.argsort(-(ideal%1),kind="stable")
    base[order[:rest]]+=1
    freq=np.zeros(alphabet,dtype=int);freq[active]=base
    return freq.tolist()


def encode(values,alphabet):
    freq=frequencies(values,alphabet)
    if not freq: return b"",[]
    starts=np.cumsum([0]+freq[:-1]).tolist()
    threshold=[((LOW//TOTAL)<<8)*f for f in freq]
    state=LOW;rev=bytearray()
    for v in reversed(values.tolist()):
        while state>=threshold[v]:
            rev.append(state&255);state>>=8
        f=freq[v];state=(state//f)*TOTAL+state%f+starts[v]
    return struct.pack("<I",state)+rev[::-1],freq


def decode(data,freq,count):
    if not count and not freq and not data: return np.empty(0,dtype=np.uint16)
    if len(data)<4 or sum(freq)!=TOTAL or min(freq)<0: raise ValueError("invalid frequency frame")
    lut=[i for i,f in enumerate(freq) for _ in range(f)]
    starts=np.cumsum([0]+freq[:-1]).tolist()
    state,=struct.unpack_from("<I",data)
    vals=np.empty(count,dtype=np.uint16)
    pos=4
    for i in range(count):
        slot=state&(TOTAL-1);sym=lut[slot];vals[i]=sym
        state=freq[sym]*(state//TOTAL)+slot-starts[sym]
        while state<LOW:
            if pos==len(data):raise ValueError("truncated vector stream")
            state=(state<<8)|data[pos];pos+=1
    if state!=LOW or pos!=len(data):raise ValueError("vector stream termination")
    return vals


def assign(vectors,book):
    result=np.empty(len(vectors),dtype=np.int32)
    c=book.astype(np.float32);norm=(c*c).sum(axis=1)
    for start in range(0,len(vectors),1024):
        x=vectors[start:start+1024].astype(np.float32)
        result[start:start+len(x)]=np.argmin(norm-2*(x@c.T),axis=1)
    return result


def fit(matrix,width,size):
    vectors=matrix.reshape(-1,width)
    # The original signed integers often repeat. Fit the unique vectors with
    # multiplicities, keeping codewords on the release's integer lattice.
    unique,inverse,count=np.unique(vectors,axis=0,return_inverse=True,return_counts=True)
    inverse=inverse.reshape(-1)
    order=np.argsort(-count,kind="stable")
    book=unique[order[:size]].copy()
    for epoch in range(5):
        labels=assign(unique,book)
        mass=np.bincount(labels,weights=count,minlength=len(book))
        total=np.stack([np.bincount(labels,weights=count*unique[:,j],minlength=len(book))
                        for j in range(width)],axis=1)
        new=book.copy();used=mass>0
        new[used]=np.rint(total[used]/mass[used,None]).astype("i1")
        new=np.unique(new,axis=0)
        if np.array_equal(book,new):break
        book=new
    labels=assign(unique,book)[inverse]
    return book,labels


def segment(q,book,labels,scale,cov,fraction):
    width=book.shape[1];original=q.reshape(-1,width)
    predicted=book[labels]
    residual=original.astype(np.float32)-predicted
    imperfect=np.flatnonzero(np.any(residual!=0,axis=1))
    escaped=np.zeros(len(labels),dtype=bool)
    if len(imperfect) and fraction<1:
        # Covariance couples the coordinates within each short vector. Rank
        # changes by their measured output error times the physical row scale.
        cols=q.shape[1]//width
        blocks=np.stack([cov[i*width:(i+1)*width,i*width:(i+1)*width] for i in range(cols)]).astype("f")
        diff=residual.reshape(len(q),cols,width)
        cost=np.einsum("rcd,cdk,rck->rc",diff,blocks,diff)*np.square(scale[:,None])
        order=imperfect[np.argsort(cost.ravel()[imperfect],kind="stable")]
        escaped[order[round(len(order)*fraction):]]=True
    ids=labels.copy();ids[escaped]=len(book)
    raw=(original[escaped].reshape(-1).astype(np.int16)+7).astype("u1")
    encoded,fi=encode(ids,len(book)+1); exceptions,fr=encode(raw,15)
    actual=predicted.copy();actual[escaped]=original[escaped]
    desc=dict(shape=list(q.shape),book=book.tolist(),indices=len(encoded),escapes=len(exceptions),
              fi=fi,fr=fr)
    return desc,encoded+exceptions,actual.reshape(q.shape),int((~escaped[imperfect]).sum())


def pack(base,parts):
    entries={name:v[0] for name,v in parts.items()}
    head=zlib.compress(json.dumps(entries,separators=(",",":")).encode(),9)
    return MAGIC+struct.pack("<II",len(base),len(head))+base+head+b"".join(v[1] for v in parts.values())


def unpack(data):
    if data[:8]!=MAGIC: raise ValueError("unexpected vector-weight format")
    n,h=struct.unpack_from("<II",data,8)
    base=data[16:16+n];cursor=16+n+h
    desc=json.loads(zlib.decompress(data[16+n:cursor]))
    restored={}
    for name,d in desc.items():
        book=np.array(d["book"],dtype="i1")
        rows,cols=d["shape"];width=book.shape[1]
        if width<1 or cols%width:raise ValueError("invalid vector geometry")
        j=cursor+d["indices"];k=j+d["escapes"]
        ids=decode(data[cursor:j],d["fi"],rows*cols//width)
        escaped=ids==len(book)
        if np.any(ids>len(book)):raise ValueError("invalid vector index")
        raw=decode(data[j:k],d["fr"],int(escaped.sum())*width)
        v=np.empty((len(ids),width),dtype="i1")
        v[~escaped]=book[ids[~escaped]];v[escaped]=(raw.astype("i1")-7).reshape(-1,width)
        restored[name]=v.reshape(rows,cols);cursor=k
    if cursor!=len(data):raise ValueError("trailing vector bytes")
    return base,restored


if __name__=="__main__":
    p=argparse.ArgumentParser(description="Decode a complete research rANS/VQ container to raw tensors")
    p.add_argument("container",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args()
    if a.output.exists(): raise SystemExit("choose a fresh output path")
    import tempfile
    base,matrices=unpack(a.container.read_bytes())
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"base.c5";path.write_bytes(base);bank=np2.read_container(str(path))
    for t in bank:
        if t.name in matrices:
            if np.any(t.a) or t.a.shape!=matrices[t.name].shape:raise ValueError("incorrect NP2 placeholder")
            t.a=matrices.pop(t.name)
    if matrices:raise ValueError("unknown matrix")
    np2.write_raw(bank,a.output)
    print(hashlib.sha256(a.output.read_bytes()).hexdigest())
