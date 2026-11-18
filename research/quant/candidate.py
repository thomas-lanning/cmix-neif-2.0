"""Orient shared MQA channels consistently in Q/K and V/output weight banks."""
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(os.environ.get("NEIF_SOURCE_ROOT",Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(ROOT/"training/tf/3_train/export/np2"))
import np2_container as np2


def orientation(pieces,shrink):
    # pieces contains one (possibly non-contiguous) block per channel and tensor.
    # A toggle flips the signed symbols in all linked blocks together.
    count=len(pieces[0])
    bins=np.array([[np.bincount(block.ravel().astype("i2")+7,minlength=15) for block in bank]
                   for bank in pieces],dtype=float)
    # Start with the largest contribution aligned, then price each toggle
    # against the resulting separate marginal histogram of each tensor.
    flip=(bins[0] @ np.arange(-7,8))<0
    for _ in range(25):
        current=np.where(flip[None,:,None],bins[:,:,::-1],bins)
        hist=current.sum(1)+.5
        prices=-np.log2(hist/hist.sum(1,keepdims=True))
        gain=((current-current[:,:,::-1])*prices[:,None,:]).sum((0,2))
        order=np.flatnonzero(gain>shrink)
        if not len(order):break
        flip[order[gain[order].argmax()]]^=True
    if len(flip)!=count:raise ValueError("channel grouping failed")
    return flip


def export(source,output,choice):
    if int(choice)!=choice or int(choice) not in range(1,7):raise ValueError("choose 1 through 6")
    tag=int(choice);mode=(tag-1)%3;threshold=0 if tag<=3 else 3
    tensors=np2.read_container(str(source))
    by_name={t.name:t for t in tensors}
    changed=0
    for idx,layer in enumerate((3,7,11,15)):
        prefix=f"blocks.{layer}.attention."
        def matrix(name):return by_name[prefix+name+"_projection.weight.q"].m2()
        q,k,v,o=(matrix(n) for n in ("query","key","value","output"))
        table=by_name[f"value_embeddings.{idx}.inner.weight.q"].m2()
        if q.shape!=(192,192) or k.shape!=(64,192) or v.shape!=k.shape or o.shape!=q.shape or table.shape[1]!=64:
            raise ValueError("unsupported attention geometry")
        if mode in (0,2):
            # Both elements of each rotary pair and all three queries share
            # their K sign. The upstream AVX kernel pairs consecutive coordinates.
            groups=np.arange(192).reshape(3,32,2).transpose(1,0,2).reshape(32,6)
            key_groups=np.arange(64).reshape(32,2)
            flip=orientation(([q[g] for g in groups],[k[g] for g in key_groups]),threshold)
            q[groups[flip].ravel()]*=-1;k[key_groups[flip].ravel()]*=-1
            changed+=int(flip.sum())
        if mode in (1,2):
            groups=np.arange(192).reshape(3,64).T
            flip=orientation(([v[i] for i in range(64)],[o[:,g] for g in groups],[table[:,i] for i in range(64)]),threshold)
            v[flip]*=-1;o[:,groups[flip].ravel()]*=-1;table[:,flip]*=-1
            changed+=int(flip.sum())
    data,_,ordered=np2.encode_np(tensors,True,np2.np_groups(tensors,True,True),True)
    with output.open('xb') as f:f.write(data)
    if not all(a.same(b) for a,b in zip(ordered,np2.read_container(str(output)),strict=True)):
        raise ValueError("stored orientation differs")
    return dict(changed_channel_groups=changed,mode=mode,histogram_threshold_bits=threshold)


if __name__=="__main__":
    print(export(Path(sys.argv[1]),Path(sys.argv[2]),float(sys.argv[3])),flush=True)
