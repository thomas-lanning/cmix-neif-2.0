"""Replace selected NP2 integer matrices with rANS while preserving the neuron permutation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib

import numpy as np
from weight_container import decode, encode, np2

MAGIC = b"NEIFHYB2"
HEADER = struct.Struct("<5I")


def labels(values, axis, groups):
    width = values.shape[1-axis]
    if groups == 1:
        return np.zeros(width, dtype=np.uint8)
    energy = np.abs(values).mean(axis=axis)
    cuts = np.quantile(energy, np.arange(1, groups) / groups)
    return np.searchsorted(cuts, energy).astype(np.uint8)


def design(rows, nr, nc):
    r = labels(rows, 1, nr)
    c = labels(rows, 0, nc)
    groups = r[:, None] * nc + c[None, :]
    source = rows.astype(np.int16) + 7
    hist = np.bincount((groups.astype(np.int32)*15 + source).ravel(),
                       minlength=nr*nc*15).reshape(-1, 15)
    tables, estimate = [], 0.
    for counts in hist:
        if not counts.sum():
            tables.append([0]*15)
            continue
        # Compute the normalized table without expanding repeated symbols.
        scale = 4096 - np.count_nonzero(counts)
        freq, rem = np.divmod(counts*scale, counts.sum())
        freq += counts != 0
        for j in np.argsort(-rem, kind="stable")[:4096-int(freq.sum())]:
            freq[j] += 1
        used = counts != 0
        estimate -= np.dot(counts[used], np.log2(freq[used]/4096)) / 8
        tables.append(freq.tolist())
    side = (r.tobytes() if nr>1 else b"") + (c.tobytes() if nc>1 else b"")
    side = zlib.compress(side + np.asarray(tables, dtype="<u2").tobytes(), 9)
    # Four bytes for the rANS state and four for the frame length.
    nframes = np.count_nonzero(hist.sum(axis=1))
    return estimate+6+len(side)+8*nframes, (nr,nc,side,groups,tables)


def pack_matrix(rows):
    configs = [(1,1),(2,1),(4,1),(8,1),(1,4),(2,2),(4,2),(2,4),(4,4)]
    _, choice = min((design(rows, r, c) for r,c in configs), key=lambda item:item[0])
    nr,nc,side,groups,tables = choice
    result = bytearray(struct.pack("<BBI",nr,nc,len(side)) + side)
    for group,freq in enumerate(tables):
        if not any(freq):
            continue
        block = encode((rows[groups==group]+7).astype(np.uint8), freq)
        result += struct.pack("<I",len(block)) + block
    return bytes(result)


def native(codec, mode, data, shape):
    if codec is None:
        raise ValueError("the adaptive matrix stream requires --matrix-tool")
    with tempfile.TemporaryDirectory() as scratch:
        src=Path(scratch)/"input"
        dst=Path(scratch)/"output"
        src.write_bytes(data)
        subprocess.run([codec,mode,str(shape[0]),str(shape[1]),src,dst],check=True)
        return dst.read_bytes()


def unpack_matrix(frame, shape, codec=None):
    if frame and frame[0]==0:
        raw=native(codec,"decode",frame[1:],shape)
        return np.frombuffer(raw,dtype=np.int8).reshape(shape)
    nr,nc,n = struct.unpack_from("<BBI",frame)
    if nr not in (1,2,4,8) or nc not in (1,2,4):
        raise ValueError("invalid group count")
    side = io.BytesIO(zlib.decompress(frame[6:6+n]))
    def index(size, groups):
        if groups==1:
            return np.zeros(size,dtype=np.uint8)
        v=np.frombuffer(side.read(size),dtype=np.uint8)
        if len(v)!=size or np.any(v>=groups):
            raise ValueError("invalid cluster label")
        return v
    groups=index(shape[0],nr)[:,None]*nc+index(shape[1],nc)[None,:]
    tables=np.frombuffer(side.read(),dtype="<u2").reshape(nr*nc,15)
    out=np.empty(shape,dtype=np.int8)
    pos=6+n
    for group,freq in enumerate(tables):
        mask=groups==group
        count=int(mask.sum())
        if not count:
            if freq.any():
                raise ValueError("unassigned distribution")
            continue
        size,=struct.unpack_from("<I",frame,pos)
        pos+=4
        out[mask]=decode(frame[pos:pos+size],count,freq.tolist())
        pos+=size
    if pos!=len(frame):
        raise ValueError("unused matrix data")
    return out


class ExternalValues:
    """Read the ordinary NP2 header and synthesize the external matrix's placeholder."""
    def __init__(self, decoder, zero_rows):
        self.decoder=decoder
        self.zero_rows=zero_rows

    def decode_ctree(self, tree, nbits, limit):
        if tree is self.zero_rows:
            return 1
        return self.decoder.decode_ctree(tree,nbits,limit)

    def __getattr__(self, name):
        return getattr(self.decoder,name)


def observe(models, tensor):
    # Later arithmetic tensors retain the original learned width and zero-row
    # distributions. The bytes of this auxiliary encoder are discarded.
    np2.code_values(np2.wc.BinEncoder(),models,tensor,np2.wc.ENC_INT4W,"")


def read_a(data, count, frames=None, codec=None):
    dec=np2.wc.BinDecoder(b"\0"+data)
    _,models,template=np2.new_state()
    known={}
    for i in range(count):
        start=dec.pos
        external=frames is not None and i in frames
        reader=ExternalValues(dec,models.zrow) if external else dec
        tensor,flags=np2.decode_tensor(reader,models,template,known)
        if external:
            if flags & ~np2.FLAG_MASK != np2.wc.ENC_INT4W or tensor.a.ndim!=2:
                raise ValueError("unexpected tensor in external directory")
            tensor=np2.Ent(tensor.name,tensor.code,unpack_matrix(frames.pop(i),tensor.a.shape,codec))
            observe(models,tensor)
        yield i,tensor,flags,dec.pos-start


def write_tensor(enc, models, template, t, flags):
    name=np2.code_head(enc,models,template,t,flags)
    np2.code_values(enc,models,t,flags & ~np2.FLAG_MASK,name)


def pack(raw, model="static", codec=None):
    if raw[:8]!=np2.MAGIC_NP2:
        raise ValueError("an NP2 container is required")
    total,count,alen,carried=struct.unpack_from("<4I",raw,8)
    tail=raw[24+alen:]
    frames,choices=bytearray(),[]
    enc,models,template=np2.new_state()
    for i,t,flags,cost in read_a(raw[24:24+alen],count):
        if flags & ~np2.FLAG_MASK == np2.wc.ENC_INT4W and t.a.ndim==2 and t.a.size>=1024:
            options=[]
            if model in ("static","mixed"):
                options.append(pack_matrix(t.m2()))
            if model in ("adaptive","mixed"):
                options.append(b"\0"+native(codec,"encode",t.a.tobytes(),t.a.shape))
            block=min(options,key=len)
            # The original cost includes the tensor header. Leave room for
            # that header as well as the six-byte directory entry.
            if len(block)+16 < cost:
                recovered=unpack_matrix(block,t.a.shape,codec)
                if not np.array_equal(recovered,t.a):
                    raise ValueError(f"rANS matrix mismatch for {t.name}")
                choices.append(dict(tensor=t.name,input_bytes=cost,rans_frame_bytes=len(block),
                                    coding="adaptive" if block[0]==0 else "static"))
                frames+=struct.pack("<HI",i,len(block))+block
                np2.code_head(enc,models,template,t,flags)
                observe(models,t)
                continue
        write_tensor(enc,models,template,t,flags)
    arithmetic=enc.finish()[1:]
    result=MAGIC+HEADER.pack(total,count,carried,len(arithmetic),len(frames))+arithmetic+frames+tail
    return result,dict(original_bytes=len(raw),hybrid_bytes=len(result),carried_permutation_bytes=carried,
                       replaced_tensors=len(choices),rans_section_bytes=len(frames),
                       arithmetic_section_bytes=len(arithmetic),retained_np2_tail_bytes=len(tail),choices=choices)


def unpack(data,codec=None):
    if data[:8]!=MAGIC:
        raise ValueError("invalid hybrid magic")
    total,count,carried,alen,flen=HEADER.unpack_from(data,8)
    pos=8+HEADER.size
    arithmetic=data[pos:pos+alen]
    pos+=alen
    end=pos+flen
    if end>len(data):
        raise ValueError("incomplete hybrid sections")
    frames={}
    while pos<end:
        i,n=struct.unpack_from("<HI",data,pos)
        pos+=6
        if i in frames or i>=count or pos+n>end:
            raise ValueError("invalid matrix directory")
        frames[i]=data[pos:pos+n]
        pos+=n
    enc,models,template=np2.new_state()
    for i,t,flags,_ in read_a(arithmetic,count,frames,codec):
        write_tensor(enc,models,template,t,flags)
    original=enc.finish()[1:]
    return np2.MAGIC_NP2+struct.pack("<4I",total,count,len(original),carried)+original+data[end:]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("pack","unpack"))
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--matrix-model",choices=("static","adaptive","mixed"),default="static")
    parser.add_argument("--matrix-tool",type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("choose a fresh output filename")
    data=args.input.read_bytes()
    tool=args.matrix_tool.resolve() if args.matrix_tool else None
    if args.action=="pack":
        result,report=pack(data,args.matrix_model,tool)
        if unpack(result,tool)!=data:
            raise ValueError("the original NP2 file was not recovered exactly")
        report.update(exact_np2_roundtrip=True,hybrid_sha256=hashlib.sha256(result).hexdigest(),
                      original_sha256=hashlib.sha256(data).hexdigest())
        print(json.dumps(report),flush=True)
    else:
        result=unpack(data,tool)
    with args.output.open("xb") as f:
        f.write(result)
