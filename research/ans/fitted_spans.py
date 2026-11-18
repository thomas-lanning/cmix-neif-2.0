"""Fit compact integer corrections to authenticated final-bit probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from numba import njit
from entropy_spans import BLOCKS, run as verify_native

TOTAL=1<<32
EVENT=np.dtype([('p','<u4'),('head','<u4'),('prefix','<u2'),('previous','u1'),('bit','u1')])


def trace(path,expected):
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=expected:raise ValueError("trace hash mismatch")
    meta=struct.unpack_from('<III',blob)
    return meta,np.frombuffer(blob,dtype=EVENT,offset=12)


def inputs(ev):
    p=ev['p'].astype(np.int64)
    prob=p/TOTAL
    confidence=np.minimum(8,np.floor(-np.log2(np.minimum(prob,1-prob)))).astype('i4')
    depth=np.floor(np.log2(np.maximum(ev['prefix'],1))).astype('i4')&7
    x=np.empty((len(p),3),dtype='i2')
    x[:,0]=256
    x[:,1]=((2*(prob>=.5).astype('i4')-1)*confidence*64)
    # Normalize the observed change between the pre-floor and final odds.
    x[:,2]=np.rint(256*np.clip((ev['head'].astype(np.int64)-p)/TOTAL/np.maximum(prob*(1-prob),1e-10),-4,4))
    return prob,depth,confidence,x


def groups(scope,depth,confidence):
    if scope=='global':return np.zeros_like(depth),1
    if scope=='depth':return depth,8
    if scope=='confidence':return np.minimum(confidence//2,3),4
    return depth*4+np.minimum(confidence//2,3),32


def fit(traces,scope,penalty):
    moment=None
    for _,events in traces:
        ev=events[::4];ev=ev[ev['prefix']!=0]
        p,depth,conf,x=inputs(ev);x=x.astype('f8')/256
        ids,n=groups(scope,depth,conf)
        if moment is None:moment=[[np.zeros((3,3)),np.zeros(3)] for _ in range(n)]
        for g in range(n):
            match=ids==g;y=ev['bit'][match];v=p[match];a=x[match]
            moment[g][0]+=(a.T*(v*(1-v)))@a
            moment[g][1]+=a.T@(y-v)
    return np.array([np.clip(np.rint(256*np.linalg.solve(h+np.eye(3)*penalty,b)),-96,96)
                     for h,b in moment],dtype='i1')


def predict(ev,scope,table):
    p,depth,conf,x=inputs(ev)
    ids,_=groups(scope,depth,conf)
    step=np.einsum('ij,ij->i',x.astype('i4'),table[ids].astype('i4')).clip(-32768,32768)
    step[ev['prefix']==0]=0
    original=ev['p'].astype('u8')
    slope=(original*(np.uint64(TOTAL)-original))>>np.uint64(32)
    # Keep all rounding integer and symmetric around zero.
    move=slope.astype('i8')*step
    return np.clip(original.astype('i8')+np.sign(move)*(abs(move)//65536),1,TOTAL-1).astype('u4')


@njit
def coder(low,high,prob,bit):
    lo,hi=np.uint64(low),np.uint64(high)
    size=0
    for i in range(len(bit)):
        mid=lo+(((hi-lo)*np.uint64(prob[i]))>>np.uint64(32))
        if bit[i]:hi=mid
        else:lo=mid+np.uint64(1)
        while (lo>>np.uint64(24))==(hi>>np.uint64(24)):
            size+=1;lo=(lo<<np.uint64(8))&np.uint64(0xffffffff);hi=((hi<<np.uint64(8))&np.uint64(0xffffffff))|np.uint64(255)
    return size


def run(a):
    sha={name: h for h,name in (s.split() for s in (a.data/'streams.sha256').read_text().splitlines())}
    oracle=a.data/'oracle-prefix.bin'
    trials={}
    baseline={}
    for b in BLOCKS:
        file=a.data/f'events-{b}.bin'
        meta,ev=trace(file,sha[file.name]);trials[b]=(meta,ev)
        measured=verify_native(a.executable.resolve(),file,oracle,(0,1,1))[(0,1,1)]
        if coder(*meta[:2],ev['p'],ev['bit'])!=measured['base_bytes']:raise ValueError("native arithmetic control differs")
        baseline[b]=measured['base_bytes']
    train=[trials[b] for b in BLOCKS[:2]]
    def score(b,scope,t):
        meta,ev=trials[b]
        return int(coder(*meta[:2],predict(ev,scope,t),ev['bit']))-baseline[b]
    select=BLOCKS[2];choices=[(0,0,'global',0,np.zeros((1,3),'i1'))]
    for scope in ('global','depth','confidence','depth-confidence'):
        for penalty in (32,256,2048):
            t=fit(train,scope,penalty)
            choices.append((score(select,scope,t),t.nbytes,scope,penalty,t))
    winner=min(choices,key=lambda r:r[:4])
    delta,size,scope,penalty,t=winner
    frozen=dict(scope=scope,penalty=penalty,table=t.tolist(),stored_coefficient_bytes=size,
                selection_block=select,selection_delta_bytes=delta,training_blocks=BLOCKS[:2])
    a.out.mkdir(exist_ok=True);(a.out/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    result=dict(frozen=frozen,nominees=[r[:4] for r in choices],baseline=baseline,
                heldout={b:score(b,scope,t) for b in BLOCKS[3:]},trace_hashes=sha)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('executable','data','out'):p.add_argument(k,type=Path)
    run(p.parse_args())
