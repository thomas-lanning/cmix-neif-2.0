"""Measure lossless rANS repacks of selected screened NP2 candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from weight_container import np2, pack, unpack


def select(directory, count):
    trials=[json.loads(p.read_bytes()) for p in directory.glob("*.json")]
    if not trials: return []
    # Cover the leading validation alternatives and the biggest changes to the
    # alphabet; a worse-performing model can still diagnose a format penalty.
    quality=sorted(trials,key=lambda r:max(r["search_proxy_bytes"],r["confirmation_proxy_bytes"]))
    width=sorted(trials,key=lambda r:r["twice_weight_delta"])
    coarse=[r for r in width if any(s["kind"] in ("rows","quant") for s in r["specs"])]
    pick=quality[:count//2]+width[:count//4]+coarse[:max(1,count//4)]
    return list({r["key"]:r for r in pick}.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("screen",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--count",type=int,default=12)
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    for r in select(a.screen,a.count):
        key=r["key"]
        target=a.out/(key+".json")
        if target.exists(): continue
        model=a.screen/(key+".c5")
        if hashlib.sha256(model.read_bytes()).hexdigest()!=r["candidate_sha256"]:
            raise ValueError("source changed")
        weights=np2.read_container(str(model))
        encoded=pack(weights, row_model=True)
        back=unpack(encoded)
        if len(weights)!=len(back) or not all(t.same(u) for t,u in zip(weights,back)):
            raise ValueError("decoded weights differ")
        delta=2*(len(encoded)-r["candidate_bytes"])
        result=dict(key=key,model_sha256=r["candidate_sha256"],np2_bytes=r["candidate_bytes"],
                    row_rans_bytes=len(encoded),repack_sha256=hashlib.sha256(encoded).hexdigest(),
                    twice_repack_delta=delta,search_proxy_before_reader=r["search_proxy_bytes"]+delta,
                    confirmation_proxy_before_reader=r["confirmation_proxy_bytes"]+delta,time_unix=time.time())
        tmp=target.with_suffix(".tmp")
        tmp.write_text(json.dumps(result,indent=2)+"\n")
        tmp.replace(target)
        print(key,result["twice_repack_delta"],round(result["confirmation_proxy_before_reader"]),flush=True)


if __name__=="__main__":
    main()
