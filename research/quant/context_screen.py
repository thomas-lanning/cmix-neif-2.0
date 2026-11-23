"""Screen article context corrections from cached native head predictions."""
import argparse
from functools import partial
import json
from pathlib import Path
import time

import numpy as np

from evaluate import article_pool, FULL_TARGETS, REFERENCE_SHA
from profile_cache import file_sha, ProfileCache

VOCAB = 208
CONTEXTS = ("top_prediction", "current_token", "hidden_cluster")
GROUPS = (208, 216, 64)


def read_article(cache, suite, window):
    output, receipt, _ = cache.get(suite, window)
    raw = (suite / window["file"]).read_bytes()
    if raw[:8] != b"NEIFQ001":
        raise ValueError("unknown fixture format")
    rows = np.frombuffer(raw, np.uint8, offset=8).reshape(-1, 424)
    scored = rows[rows[:, 6] == 1]
    y = scored[:, 5].astype(int)
    features = np.memmap(output, dtype="<f4", mode="r", shape=(len(y), 400))
    p = features[:, 192:].astype(np.float64)
    if (
        len(y) != receipt["targets"]
        or not np.isfinite(features).all()
        or p.min() < 0
        or np.max(abs(p.sum(1) - 1)) >= 0.002
    ):
        raise ValueError("invalid native head output")
    loss = -np.log2(np.maximum(3e-8, p[np.arange(len(y)), y])).sum()
    if abs(loss - receipt["bits"]) >= 1e-4 * len(y):
        raise ValueError("loss differs from native replay")
    p /= p.sum(1, keepdims=True)
    return p, y, features[:, :192], scored[:, 0].astype(int)


def profile(cache, suite, window):
    p, y, hidden, current = read_article(cache, suite, window)
    h = hidden[:, :32]
    k = abs(h).argmax(1)
    contexts = (p.argmax(1), current, 2 * k + (h[np.arange(len(y)), k] > 0))
    return p, y, contexts


def fit_article(cache, suite, window):
    p, y, contexts = profile(cache, suite, window)
    updates = []
    for groups, ids in zip(GROUPS, contexts):
        counts = np.bincount(ids * VOCAB + y, minlength=groups * VOCAB).reshape(groups, VOCAB)
        # Collate in document order within each context. Absent contexts need no update.
        order = np.argsort(ids, kind="stable")
        keys, offsets = np.unique(ids[order], return_index=True)
        updates.append((counts, keys, np.add.reduceat(p[order], offsets, axis=0)))
    return updates


def article_scores(cache, suite, models, window):
    p, y, contexts = profile(cache, suite, window)
    observed = p[np.arange(len(y)), y]
    original = np.maximum(3e-8, observed)
    deltas = []
    for j, _, _, table in models:
        factors = 1 + table[contexts[j]].astype(np.float64) / 128
        revised = observed * factors[np.arange(len(y)), y] / (p * factors).sum(1)
        deltas.append(float(np.log2(original / np.maximum(3e-8, revised)).sum()))
    return len(y), deltas


def ordered_batches(pool, fn, items, size):
    # Bound queued per-article arrays even if one earlier article takes longer.
    for start in range(0, len(items), size):
        yield from pool.map(fn, items[start : start + size])


def run(args):
    train = args.data / "training"
    hold = args.data / "sealed96"
    if file_sha(args.model) != REFERENCE_SHA:
        raise ValueError("the control must be the published model")
    ws = json.loads((train / "manifest.json").read_bytes())["windows"]
    hs = json.loads((hold / "manifest.json").read_bytes())["windows"]
    if (
        len(ws) != 1024
        or len(hs) != 96
        or len({w["document"] for w in ws + hs}) != len(ws) + len(hs)
    ):
        raise ValueError("training, selection and held-out articles must be disjoint")
    fit = [w for i, w in enumerate(ws) if i % 8 != 4]
    tune = ws[4::8]
    cache = ProfileCache(args.cache, args.exe, args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / "status"
    start = time.monotonic()
    batch = 2 * len(args.cores)
    counts = [np.zeros((g, VOCAB)) for g in GROUPS]
    expected = [np.zeros_like(a) for a in counts]
    with article_pool(args.cores) as pool:
        status.write_text("fitting\n")
        for updates in ordered_batches(pool, partial(fit_article, cache, train), fit, batch):
            for j, (observed, keys, sums) in enumerate(updates):
                counts[j] += observed
                expected[j][keys] += sums
        models = []
        for j in range(len(CONTEXTS)):
            for shrink in (8, 64):
                ratio = (counts[j] + shrink) / (expected[j] + shrink)
                for strength in (0.25, 0.5, 1.0):
                    table = np.rint(128 * (ratio**strength - 1)).clip(-64, 127).astype("int8")
                    models.append((j, shrink, strength, table))
        fitted = time.monotonic()
        status.write_text("selecting\n")
        n = 0
        totals = np.zeros(len(models))
        for length, deltas in ordered_batches(
            pool, partial(article_scores, cache, train, models), tune, batch
        ):
            n += length
            totals += deltas
        ranks = totals / n * FULL_TARGETS / 8 + np.array([2 * m[3].nbytes for m in models])
        best = int(np.argmin(ranks))
        j, shrink, strength, table = models[best]
        table_file = args.output / "frozen-table.i8"
        table.tofile(table_file)
        frozen = dict(
            model=dict(name=CONTEXTS[j], j=j, shrink=shrink, strength=strength),
            table_bytes=table.nbytes,
            table_sha256=file_sha(table_file),
            tuning_targets=n,
            tuning_delta_bits=float(totals[best]),
            tuning_proxy_with_table_bytes=float(ranks[best]),
        )
        (args.output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
        selected = time.monotonic()
        status.write_text("testing frozen choice\n")
        fn = partial(article_scores, cache, hold, [models[best]])
        articles = [
            dict(document=w["document"], targets=length, delta_bits=deltas[0])
            for w, (length, deltas) in zip(hs, ordered_batches(pool, fn, hs, batch), strict=True)
        ]
    done = time.monotonic()
    n = sum(w["targets"] for w in articles)
    bits = sum(w["delta_bits"] for w in articles)
    # Sample whole documents in each of eight bands to retain broad corpus coverage.
    positions = {w["document"]: w["stream_byte"] for w in hs}
    bands = np.array_split(sorted(articles, key=lambda w: positions[w["document"]]), 8)
    rng = np.random.default_rng(23)
    samples = np.zeros((2000, 2))
    for band in bands:
        values = np.array([[w["delta_bits"], w["targets"]] for w in band])
        samples += values[rng.integers(len(values), size=(len(samples), len(values)))].sum(1)
    ci = np.quantile(samples[:, 0] / samples[:, 1], [0.025, 0.975]).tolist()
    result = dict(
        frozen=frozen,
        fit_articles=len(fit),
        selection_articles=len(tune),
        heldout_articles=len(hs),
        heldout_targets=n,
        heldout=articles,
        heldout_primary_head_bits_per_target_delta=bits / n,
        article_bootstrap_95=ci,
        estimated_two_table_copies_bytes=2 * table.nbytes,
        stage_seconds=dict(fit=fitted - start, select=selected - fitted, test=done - selected),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    status.write_text("complete\n")
    print(json.dumps({k: v for k, v in result.items() if k != "heldout"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "exe", "model", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--cores", required=True, nargs="+", type=int)
    run(parser.parse_args())
