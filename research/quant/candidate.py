"""Reduce BF16 mantissa precision in row scales and auxiliary full-precision tensors."""
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(os.environ.get("NEIF_SOURCE_ROOT",Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(ROOT/"training/tf/3_train/export/np2"))
import np2_container as np2


def belongs(t,family):
    if family==1:
        return t.name.endswith("mlp.up.weight.scale") or t.name.endswith("mlp.down.weight.scale")
    if family==2:
        return t.name.endswith("_convolution.weight")
    if family==3:
        return t.name.endswith((".q_mix.w",".q_mlp.w",".q_final.w"))
    if family==4:
        return t.a.dtype==np.uint16 and t.name.endswith("weight.scale")
    if family==5:
        return any(belongs(t,f) for f in (2,3,4))
    raise ValueError("unsupported tensor family")


def rounded(raw,drop):
    # Ties round to the even bin. Widen before addition to preserve the exponent carry.
    u=raw.astype(np.uint32)
    return (((u+(1<<(drop-1))-1+((u>>drop)&1))>>drop)<<drop).astype(np.uint16)


def export(source,output,spec):
    if not float(spec).is_integer(): raise ValueError("expected family and precision encoded as two digits")
    family,drop=divmod(int(spec),10)
    if family not in (1,2,3,4,5) or not 1<=drop<=4:raise ValueError("unsupported precision choice")
    bank=np2.read_container(str(source))
    changed=0
    for t in bank:
        if not belongs(t,family): continue
        a=t.a
        if a.dtype==np.uint16:
            raw=a
        elif a.dtype==np.float32 and not np.any(a.view(np.uint32)&65535):
            raw=(a.view(np.uint32)>>16).astype(np.uint16)
        else:raise ValueError(f"expected BF16 parameter in {t.name}")
        result=rounded(raw,drop)
        if np.any((result&0x7f80)==0x7f80): raise ValueError("rounding crossed finite range")
        changed+=np.count_nonzero(raw!=result)
        t.a=result if a.dtype==np.uint16 else (result.astype(np.uint32)<<16).view(np.float32)
    if not changed: raise ValueError("no parameters changed")
    blob,_,ordered=np2.encode_np(bank,True,np2.np_groups(bank,True,True),True)
    with output.open("xb") as f:f.write(blob)
    if not all(a.same(b) for a,b in zip(ordered,np2.read_container(str(output)),strict=True)):
        raise ValueError("reloaded export differs")
    return dict(family=family,remaining_mantissa_bits=7-drop,changed=int(changed))


if __name__=="__main__":
    print(export(Path(sys.argv[1]),Path(sys.argv[2]),float(sys.argv[3])),flush=True)
