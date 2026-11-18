"""Choose different quantization budgets per layer from paired native replay costs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from allocation import export, np2
from check_training_split import TRAINING_SHA
from evaluate import BASELINE, FULL_TARGETS, REFERENCE_SHA, measure
from paired_stats import summarize


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def subset(source, output, start):
    parent_sha = digest(source/"manifest.json")
    if (output/"manifest.json").exists():
        previous = json.loads((output/"manifest.json").read_bytes())
        if previous.get("parent_sha256") != parent_sha or previous.get("training_fold") != start:
            raise ValueError("existing fold belongs to another training set")
        return
    info=json.loads((source/"manifest.json").read_bytes())
    windows = info["windows"]
    if len(windows) == 96:
        chosen = windows[start::4]
    elif len(windows) == 1024 and parent_sha == TRAINING_SHA:
        # Use the quarter and three-quarter points of each of 24 equal strata.
        chosen = [windows[(4*i + start)*len(windows)//96] for i in range(24)]
    else:
        raise ValueError("expected 96 training articles or the verified full-corpus training set")
    output.mkdir(exist_ok=True)
    for w in chosen:
        src=source/w["file"];dst=output/w["file"]
        if digest(src)!=w["sha256"]:raise ValueError("fixture changed")
        if not dst.exists():shutil.copyfile(src,dst)
    info.update(windows=chosen,parent_sha256=parent_sha,training_fold=start)
    tmp=output/"manifest.tmp";tmp.write_text(json.dumps(info,indent=2)+"\n");tmp.replace(output/"manifest.json")


def run(a):
    if digest(BASELINE)!=REFERENCE_SHA:raise ValueError("baseline changed")
    bank=np2.read_container(str(BASELINE))
    a.output.mkdir(parents=True,exist_ok=True)
    sets=[a.output/f"articles-{i}" for i in (1,3)]
    for s,i in zip(sets,(1,3)):subset(a.training,s,i)
    root=a.output/"screen";root.mkdir(exist_ok=True)
    validation=a.output/"fold-2";validation.mkdir(exist_ok=True)
    anchor=(Path(__file__).read_bytes()+a.exe.read_bytes()+(sets[0]/"manifest.json").read_bytes()+
            Path(__file__).with_name("allocation.py").read_bytes()+Path(__file__).with_name("evaluate.py").read_bytes())
    def score(specs):
        if time.time()>=a.until:raise TimeoutError
        key=hashlib.sha256(anchor+json.dumps(specs,sort_keys=True).encode()).hexdigest()[:20]
        path=root/(key+".json")
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=root) as d:
                tmp=Path(d);model=tmp/"export.c5";changed=export(bank,specs,model)
                r=measure(a.exe,sets[0],model,a.output/"discovery",key)
                r.update(key=key,specs=specs,changed_rows=changed)
                model.replace(root/(key+".c5"));record=tmp/"result.json";record.write_text(json.dumps(r)+"\n");record.replace(path)
                print("scored",key,r["twice_weight_delta"],round(r["search_proxy_bytes"]),round(r["confirmation_proxy_bytes"]),flush=True)
        return json.loads(path.read_bytes())
    def other(row):
        path=validation/(row["key"]+".json")
        if not path.exists():
            if time.time()>=a.until:raise TimeoutError
            result=measure(a.exe,sets[1],root/(row["key"]+".c5"),validation,row["key"])
            if result["candidate_sha256"]!=row["candidate_sha256"]:raise ValueError("held-out export changed")
            tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(result)+"\n");tmp.replace(path)
        return json.loads(path.read_bytes())
    def cost(row,alpha):
        d=np.asarray(row["bits_per_target_delta_by_window"])
        return row["twice_weight_delta"]+alpha*max(d[::2].mean(),d[1::2].mean())*FULL_TARGETS/8
    def spec(side,l,f):return dict(kind="rows",family=side,layers=[l],fraction=f,levels=3)
    rows=[]
    for layer in range(16):
        for side in ("down","up","attention"):
            rows.append(score([spec(side,layer,.125)]))
    while not a.transfer.exists():
        if time.time()>=a.until:raise TimeoutError
        time.sleep(30)
    alpha=json.loads(a.transfer.read_bytes())["ranking_transfer"]
    chosen={}
    for scale in (alpha,alpha/2,1.):
        for row in sorted(rows,key=lambda r:cost(r,scale))[:8]:chosen[row["key"]]=row
    ranked=sorted(chosen.values(),key=lambda r:max(cost(r,alpha),cost(other(r),alpha)))
    for row in ranked[:5]:
        s=row["specs"][0]
        for fraction in (.03125,.5,1.):rows.append(score([spec(s["family"],s["layers"][0],fraction)]))
    finalists={}
    for row in sorted(rows,key=lambda r:cost(r,alpha))[:15]+ranked[:6]:
        score2=max(cost(row,alpha),cost(other(row),alpha)); s=row["specs"][0]
        key=(s["family"],s["layers"][0])
        if key not in finalists or score2<finalists[key][0]:finalists[key]=(score2,row)
    order=[r for _,r in sorted(finalists.values(),key=lambda t:t[0])]
    # Complementary document errors can make a combination useful. Freeze
    # the candidates selected from the two training folds before evaluation.
    tested=list(order)
    for count in (2,4,8):
        if count>len(order):continue
        combo=score([r["specs"][0] for r in order[:count]]);other(combo);tested.append(combo)
    best=sorted(tested,key=lambda r:max(cost(r,alpha),cost(other(r),alpha)))
    selection=[dict(key=r["key"],sha256=r["candidate_sha256"],specs=r["specs"],
                    worst_training_split=max(cost(r,alpha),cost(other(r),alpha))) for r in best[:4]]
    (a.output/"selection.json").write_text(json.dumps(dict(transfer=alpha,
        transfer_source_sha256=digest(a.transfer),candidates=selection),indent=2)+"\n")
    if a.held_out:
        manifest = a.held_out / "manifest.json"
        if digest(manifest) != "4aaae03b731808a4cb09524a4f759cf120648a1d3549a160f7dd3a40bce4d85d":
            raise ValueError("unexpected 96-article evaluation set")
        held = a.output / "held-out-96"
        outcomes = []
        for candidate in selection:
            if time.time() >= a.until:
                raise TimeoutError
            model = root / (candidate["key"] + ".c5")
            if digest(model) != candidate["sha256"]:
                raise ValueError("candidate changed after selection")
            trial = measure(a.exe, a.held_out, model, held, candidate["key"])
            outcomes.append(summarize(trial, manifest, alpha))
            print("held-out", candidate["key"], outcomes[-1]["projected_entry_delta_bytes"], flush=True)
        (held/"summary.json").write_text(json.dumps(outcomes, indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("exe","training","transfer","output"):p.add_argument(n,type=Path)
    p.add_argument("--until",type=float,default=float("inf"))
    p.add_argument("--held-out",type=Path,help="replay the four frozen finalists on the independent 96 articles")
    a=p.parse_args()
    try:run(a)
    except TimeoutError:print("search deadline reached",flush=True)
