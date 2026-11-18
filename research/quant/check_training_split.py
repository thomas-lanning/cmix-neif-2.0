"""Check training article selection against its frozen source plan and held-out benchmark."""
import hashlib
import json
from pathlib import Path
import sys


def check(folder):
    def manifest(name):
        return json.loads((folder / name).read_bytes())
    def documents(data):
        return {r["document"] for r in data["windows"]}
    train = manifest("training/manifest.json")
    planned = manifest("train1024.json")
    holdouts = [manifest(f"{name}/manifest.json") for name in ("suite", "sealed96")]
    ids = documents(train)
    if len(train["windows"]) != 1024 or len(ids) != 1024 or ids != documents(planned):
        raise ValueError("training articles differ from the selected plan")
    for reference, count in zip(holdouts, (12, 96), strict=True):
        if len(documents(reference)) != count or ids & documents(reference):
            raise ValueError("unexpected holdout size or overlapping training articles")
    if documents(holdouts[0]) & documents(holdouts[1]):
        raise ValueError("two evaluation suites overlap")
    return dict(training_documents=len(ids), independent_documents=[12, 96],
                scored_targets=sum(w["targets"] for w in train["windows"]),
                training_manifest_sha256=hashlib.sha256((folder / "training/manifest.json").read_bytes()).hexdigest())


def validate_for_qat(path, quick, validation):
    raw = path.read_bytes()
    training = json.loads(raw)
    def docs(manifest):
        return {w["document"] for w in manifest["windows"]}
    if any(training[k] != quick[k] or training[k] != validation[k]
           for k in ("input_sha256", "format")):
        raise ValueError("training and evaluation use different source data")
    count = len(training["windows"])
    if len(docs(training)) != count:
        raise ValueError("repeated training document")
    if training["selection"] == quick["selection"] == validation["selection"] and count == 96:
        if docs(training) & (docs(quick) | docs(validation)):
            raise ValueError("training document overlaps early evaluation")
        return
    if (count != 1024 or training["selection"] != "full" or
            quick["selection"] != "40000000" or validation["selection"] != "40000000"):
        raise ValueError("unexpected training and evaluation geometry")
    receipt = json.loads((path.parents[1] / "split-check.json").read_bytes())
    if (receipt["training_manifest_sha256"] != hashlib.sha256(raw).hexdigest() or
            receipt["independent_documents"] != [12, 96] or receipt["training_documents"] != count):
        raise ValueError("training suite differs from its verified disjoint corpus split")
    if any(w["stream_byte"] < 45000000 for w in training["windows"]):
        raise ValueError("full-corpus training article intersects the early benchmark region")


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1])), indent=2))
