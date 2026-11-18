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


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1])), indent=2))
