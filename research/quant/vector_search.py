"""Screen a complete vector-codebook weight package using the native quantized predictor."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from allocation import bf16, np2
from evaluate import BASELINE, REFERENCE_SHA, measure

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ans"))
import vector_weights as vq


def checked_np2(tensors):
    blob,_,permuted=np2.encode_np(tensors,True,np2.np_groups(tensors,True,True),True)
    if not all(t.same(u) for t,u in zip(tensors,permuted,strict=True)):
        raise ValueError("NP2 packing changed neuron order")
    return blob


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("replay","suite","gram","output"):p.add_argument(n,type=Path)
    p.add_argument("--until",type=float,default=float("inf"))
    a=p.parse_args()
    a.replay,a.suite,a.gram,a.output=(p.resolve() for p in (a.replay,a.suite,a.gram,a.output))
    if hashlib.sha256(BASELINE.read_bytes()).hexdigest()!=REFERENCE_SHA:raise ValueError("reference differs")
    screen=a.output/"screen";screen.mkdir(exist_ok=True,parents=True)
    bank=np2.read_container(str(BASELINE))
    names={t.name:t for t in bank}
    provenance=hashlib.sha256(Path(__file__).read_bytes()+Path(vq.__file__).read_bytes()+
        (a.gram/"manifest.json").read_bytes()+(a.suite/"manifest.json").read_bytes()+a.replay.read_bytes()).hexdigest()
    # First quantify the full potential in the twelve MLPs whose widths are
    # independent. On smaller groups, measure how much the rest of the model tolerates.
    for group in (range(12),range(8,12),range(4),range(4,8)):
        layers=list(group);targets=[f"blocks.{i}.mlp.down.weight.q" for i in layers]
        if time.time()>=a.until: return
        blank=[np2.Ent(t.name,t.code,np.zeros_like(t.a) if t.name in targets else t.a) for t in bank]
        base=checked_np2(blank)
        with tempfile.TemporaryDirectory(dir=screen) as td:
            path=Path(td)/"base.c5";path.write_bytes(base)
            if not all(t.same(u) for t,u in zip(blank,np2.read_container(str(path)),strict=True)):
                raise ValueError("NP2 scaffold did not decode identically")
        cov={n:np.load(a.gram/f"layer-{layer:02d}-down.npy") for layer,n in zip(layers,targets)}
        for width,size in ((4,256),(4,1024),(2,64),(4,64)):
            if time.time()>=a.until:return
            books={n:vq.fit(names[n].a,width,size) for n in targets}
            for fraction in (0.,.02,.125,.5,1.):
                if time.time()>=a.until:return
                spec=dict(layers=layers,width=width,codewords=size,quantized_fraction=fraction)
                key=hashlib.sha256(json.dumps([provenance,spec],sort_keys=True).encode()).hexdigest()[:20]
                if (screen/(key+".json")).exists():continue
                parts={};recovered={};changes=0
                for n in targets:
                    scales=bf16(names[n[:-1]+"scale"].a).reshape(-1)
                    desc,b,q,k=vq.segment(names[n].a,*books[n],scales,cov[n],fraction)
                    parts[n]=(desc,b);recovered[n]=q;changes+=k
                packed=vq.pack(base,parts)
                unpacked,decoded=vq.unpack(packed)
                if unpacked!=base or any(not np.array_equal(recovered[n],decoded[n]) for n in targets):
                    raise ValueError("rANS/VQ decode differs")
                new=[np2.Ent(t.name,t.code,decoded.get(t.name,t.a)) for t in bank]
                if not fraction and not all(t.same(u) for t,u in zip(bank,new,strict=True)):
                    raise ValueError("lossless vector control changed tensor values")
                delta=2*(len(packed)-BASELINE.stat().st_size)
                record=dict(key=key,spec=spec,provenance=provenance,vq_bytes=len(packed),
                    vq_sha256=hashlib.sha256(packed).hexdigest(),twice_vq_delta=delta,changed_vectors=changes,
                    placeholder_bytes=len(base),native_reader_bytes_measured=False)
                # Lossless controls have identical tensors. A larger package
                # with changed values can still be tested when it saves enough
                # error to justify its cost, but this search targets weight savings.
                if not changes:
                    record.update(search_projection=delta,confirmation_projection=delta,exact_reference_tensors=True)
                elif delta<0:
                    blob=checked_np2(new)
                    with tempfile.TemporaryDirectory(dir=screen) as td:
                        path=Path(td)/"candidate.c5";path.write_bytes(blob)
                        if not all(t.same(u) for t,u in zip(new,np2.read_container(str(path)),strict=True)):
                            raise ValueError("native replay surrogate differs from vector package")
                        r=measure(a.replay,a.suite,path,a.output/"discovery",key)
                        tax=delta-r["twice_weight_delta"]
                        record.update(candidate_sha256=r["candidate_sha256"],surrogate_bytes=len(blob),
                            search_projection=r["search_proxy_bytes"]+tax,confirmation_projection=r["confirmation_proxy_bytes"]+tax,
                            bits_per_target_delta_by_window=r["bits_per_target_delta_by_window"])
                        path.replace(screen/(key+".c5"))
                else:record["reason"]="larger weight package"
                if delta<0:(screen/(key+".vq")).write_bytes(packed)
                with tempfile.NamedTemporaryFile(dir=screen,mode="w",delete=False) as f:json.dump(record,f,indent=2)
                Path(f.name).replace(screen/(key+".json"))
                print("vectors",key,changes,delta,record.get("search_projection"),record.get("confirmation_projection"),flush=True)


if __name__=="__main__":main()
