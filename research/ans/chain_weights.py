"""Carry one rANS state across selected tensor matrices in an NP2 weight package."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
import hybrid_weights as hybrid
from hybrid_weights import HEADER,np2

MAGIC=b"NEIFCHN1"
LOWER=1<<23


def matrix(tool,action,data,shape,mode,state):
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/"input";out=Path(td)/"output"
        src.write_bytes(data)
        subprocess.run([tool,action,*map(str,shape),src,out,str(mode),str(state)],check=True)
        return out.read_bytes()


def pack(source,legacy,matrix_tool,chain_tool):
    if hybrid.unpack(legacy,matrix_tool)!=source:
        raise ValueError("input hybrid and proposed source differ")
    total,count,carried,alen,flen=HEADER.unpack_from(legacy,8)
    begin=8+HEADER.size+alen
    modes={}
    pos=begin
    while pos<begin+flen:
        i,n=struct.unpack_from("<HI",legacy,pos);pos+=6
        if i in modes or legacy[pos]!=0 or legacy[pos+1]>=32 or n<6:
            raise ValueError("expected an ordered adaptive matrix package")
        modes[i]=legacy[pos+1]
        pos+=n
    _,n_a,a_len,_=struct.unpack_from("<4I",source,8)
    tensors=[(i,t) for i,t,_,_ in hybrid.read_a(source[24:24+a_len],n_a) if i in modes]
    blocks=[]
    state=LOWER
    for i,t in reversed(tensors):
        frame=matrix(chain_tool,"encode",t.a.tobytes(),t.a.shape,modes[i],state)
        state,=struct.unpack_from("<I",frame)
        blocks.append(frame[4:])
    directory=bytearray(struct.pack("<H",len(tensors)))
    prev=0
    for i,_ in tensors:
        if not 0<=i-prev<256:
            raise ValueError("tensor gap exceeds this encoding")
        directory+=bytes((i-prev,modes[i]))
        prev=i
    stream=directory+struct.pack("<I",state)+b"".join(reversed(blocks))
    return MAGIC+HEADER.pack(total,count,carried,alen,len(stream))+legacy[8+HEADER.size:begin]+stream+legacy[begin+flen:]


def unpack(data,tool):
    if data[:8]!=MAGIC:
        raise ValueError("unexpected chained package")
    total,count,carried,alen,elen=HEADER.unpack_from(data,8)
    start=8+HEADER.size
    source=data[start:start+alen]
    section=data[start+alen:start+alen+elen]
    tail=data[start+alen+elen:]
    n,=struct.unpack_from("<H",section)
    modes={}
    i=0
    for j in range(n):
        gap,mode=struct.unpack_from("<BB",section,2+j*2)
        i+=gap
        if i>=count or i in modes or mode>=32:
            raise ValueError("bad tensor directory")
        modes[i]=mode
    pos=2+2*n
    state,=struct.unpack_from("<I",section,pos);pos+=4
    dec=np2.wc.BinDecoder(b"\0"+source)
    _,models,template=np2.new_state()
    enc,original,original_template=np2.new_state()
    known={}
    for idx in range(count):
        reader=hybrid.ExternalValues(dec,models.zrow) if idx in modes else dec
        t,flags=np2.decode_tensor(reader,models,template,known)
        if idx in modes:
            if (flags&~np2.FLAG_MASK)!=np2.wc.ENC_INT4W or t.a.ndim!=2:
                raise ValueError("unexpected external tensor type")
            raw=matrix(tool,"decode",section[pos:],t.a.shape,modes[idx],state)
            state,used=struct.unpack_from("<II",raw);pos+=used
            if pos>len(section) or len(raw)!=8+t.a.size:
                raise ValueError("invalid decoded matrix size")
            t=np2.Ent(t.name,t.code,np.frombuffer(raw[8:],dtype=np.int8).reshape(t.a.shape))
            hybrid.observe(models,t)
        hybrid.write_tensor(enc,original,original_template,t,flags)
    if state!=LOWER or pos!=len(section):
        raise ValueError("rANS terminal state or cursor differs")
    original_a=enc.finish()[1:]
    return np2.MAGIC_NP2+struct.pack("<4I",total,count,len(original_a),carried)+original_a+tail


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("reference","legacy","matrix_tool","chain_tool","output"):
        p.add_argument(name,type=lambda s:Path(s).resolve())
    a=p.parse_args()
    reference=a.reference.read_bytes()
    old=a.legacy.read_bytes()
    packed=pack(reference,old,a.matrix_tool,a.chain_tool)
    if unpack(packed,a.chain_tool)!=reference:
        raise ValueError("did not independently reconstruct the underlying NP2")
    with a.output.open("xb") as f:
        f.write(packed)
    print(json.dumps(dict(original_bytes=len(reference),previous_hybrid_bytes=len(old),hybrid_bytes=len(packed),
                          hybrid_sha256=hashlib.sha256(packed).hexdigest(),original_sha256=hashlib.sha256(reference).hexdigest(),
                          exact_np2_roundtrip=True)),flush=True)
