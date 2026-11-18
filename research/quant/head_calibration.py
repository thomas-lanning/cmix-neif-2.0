"""Fit a byte-sized probability correction to native transformer predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import REFERENCE_SHA
from fixtures import MAGIC, ROW


CUTS = np.array([1/256, 1/64, 1/16, 1/4, 1/2, 3/4, 15/16], dtype=np.float32)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_documents(suite, profile):
    manifest = json.loads((suite / "manifest.json").read_bytes())
    capture = json.loads((profile / "manifest.json").read_bytes())
    if capture["model"] != REFERENCE_SHA or capture["suite"] != digest(suite / "manifest.json"):
        raise ValueError("these profiles must come from the shipped model and this suite")
    documents = []
    for w, output in zip(manifest["windows"], capture["files"], strict=True):
        window = suite / w["file"]
        features = profile / output["file"]
        if (window.stem != features.stem or digest(window) != w["sha256"] or
                digest(features) != output["sha256"]):
            raise ValueError("profile or fixture differs")
        raw = window.read_bytes()
        if raw[:len(MAGIC)] != MAGIC:
            raise ValueError("invalid fixture magic")
        rows = np.frombuffer(raw, dtype=ROW, offset=len(MAGIC))
        rows = rows[rows["score"] == 1]
        prob = np.memmap(features, dtype="<f4", mode="r").reshape(-1, 400)[:, 192:]
        if (len(rows) != len(prob) or len(rows) != w["targets"] or not np.isfinite(prob).all()
                or prob.min() < 0 or np.max(np.abs(prob.sum(1)-1)) > 1e-5):
            raise ValueError("invalid native probability vectors")
        prior = rows["prior"].view("<f2").astype(np.float32)
        confidence = np.searchsorted(CUTS, prob, side="right")
        agreement = (prior >= prob/4).astype(int) + (prior >= 4*prob).astype(int)
        groups = confidence*3 + agreement
        n = len(rows)
        mass = np.bincount((24*np.arange(n)[:, None] + groups).ravel(), weights=prob.ravel(),
                           minlength=n*24).reshape(n, 24)
        token = rows["target"].astype(int)
        mass /= mass.sum(1, keepdims=True)
        documents.append((w["document"], mass, groups[np.arange(n), token],
                          prob[np.arange(n), token].astype(np.float64)))
    return documents


def grouped(docs, scope):
    mass = np.concatenate([d[1] for d in docs])
    target = np.concatenate([d[2] for d in docs])
    if scope == "confidence":
        mass = mass.reshape(len(mass), 8, 3).sum(2)
        target = target // 3
    return mass, target


def fit(docs, scope, penalty):
    mass, target = grouped(docs, scope)
    size = mass.shape[1]
    counts = np.bincount(target, minlength=size)
    beta = np.zeros(size)
    for _ in range(16):
        pred = mass*np.exp(beta)
        pred /= pred.sum(1, keepdims=True)
        sums = pred.sum(0)
        hessian = np.diag(sums + penalty) - pred.T @ pred
        step = np.linalg.solve(hessian, counts - sums - penalty*beta)
        beta += np.clip(step, -0.25, 0.25)
        if np.max(np.abs(step)) < 1e-7:
            break
    return np.clip(np.rint(128*np.expm1(beta)), -64, 127).astype(np.int8)


def score(docs, scope, table):
    factors = 1 + np.asarray(table, dtype=np.float64)/128
    results = []
    for doc in docs:
        mass, target = grouped([doc], scope)
        final = doc[3]*factors[target]/(mass @ factors)
        original = np.maximum(doc[3], 3e-8)
        delta = np.log2(original/np.maximum(final, 3e-8))
        results.append(dict(document=doc[0], targets=len(target), delta_bits=float(delta.sum())))
    return results


def run(suite, profile, output, frozen=None):
    docs = read_documents(suite, profile)
    if len({d[0] for d in docs}) != len(docs):
        raise ValueError("repeated articles in this suite")
    if frozen:
        record = json.loads(frozen.read_bytes())
        if set(record["all_documents"]) & {d[0] for d in docs}:
            raise ValueError("external test includes an article used in the first experiment")
        scope, table = record["scope"], record["integer_table"]
        results = score(docs, scope, table)
        report = dict(frozen_sha256=digest(frozen), external=results)
    else:
        if len(docs) != 12:
            raise ValueError("the initial sweep assumes the twelve C40 windows")
        discovery, confirmation = docs[::2], docs[1::2]
        training, tuning = discovery[:4], discovery[4:]
        choices = []
        for scope in ("confidence", "confidence-and-prior"):
            for penalty in (8, 64, 512):
                table = fit(training, scope, penalty)
                rows = score(tuning, scope, table)
                choices.append((sum(r["delta_bits"] for r in rows), len(table), scope, penalty))
        best_delta, _, scope, penalty = min(choices)
        table = fit(discovery, scope, penalty) if best_delta < 0 else np.zeros(8 if scope=="confidence" else 24, "i1")
        results = score(confirmation, scope, table)
        report = dict(scope=scope, penalty=penalty, integer_table=table.tolist(), table_bytes=len(table),
                      all_documents=[d[0] for d in docs], discovery=[d[0] for d in discovery],
                      tuning_choices=choices, confirmation=results)
    report.update(suite_sha256=digest(suite/"manifest.json"), profile_sha256=digest(profile/"manifest.json"),
                  targets=sum(r["targets"] for r in results), delta_bits=sum(r["delta_bits"] for r in results))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(report["targets"], report["delta_bits"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "profile", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--frozen", type=Path)
    a = p.parse_args()
    run(a.suite, a.profile, a.output, a.frozen)
