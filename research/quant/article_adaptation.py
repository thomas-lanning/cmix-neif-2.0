"""Screen two predictors that adapt to earlier decoded blocks within an article."""
import argparse
from functools import partial
import json
from pathlib import Path
import time

import numpy as np

from context_screen import read_article, VOCAB
from evaluate import article_pool, FULL_TARGETS, REFERENCE_SHA
from profile_cache import file_sha, ProfileCache


BLOCK = 32
# No external tables. Each option needs only its two constants in the decoder.
OPTIONS = [(prior, strength) for prior in (2, 8, 32) for strength in (0.25, 0.5, 1.0)]
RETRIEVAL = [(cosine, weight) for cosine in (0.80, 0.92) for weight in (0.03125, 0.125, 0.25)]


def score(cache, suite, options, window):
    p, y, _, _ = read_article(cache, suite, window)
    starts = np.arange(0, len(y), BLOCK)
    blocks = np.arange(len(y)) // BLOCK
    # A block can only see counts from earlier complete blocks in this article.
    expected = np.zeros((len(starts), VOCAB))
    actual = np.zeros_like(expected)
    expected[1:] = np.add.reduceat(p, starts, axis=0)[:-1].cumsum(axis=0)
    counts = np.bincount(blocks * VOCAB + y, minlength=len(starts) * VOCAB).reshape(-1, VOCAB)
    actual[1:] = counts[:-1].cumsum(axis=0)
    observed = p[np.arange(len(y)), y]
    original = np.maximum(3e-8, observed)
    deltas = []
    for prior, strength in options:
        multiplier = ((actual + prior) / (expected + prior)) ** strength
        factors = np.rint(128 * multiplier).clip(64, 255)[blocks] / 128
        revised = observed * factors[np.arange(len(y)), y] / (p * factors).sum(1)
        deltas.append(float(np.log2(original / np.maximum(3e-8, revised)).sum()))
    return dict(document=window["document"], targets=len(y), delta_bits=deltas)


def retrieve(cache, suite, options, window):
    p, y, hidden, _ = read_article(cache, suite, window)
    # Fixed first 32 latent dimensions, with no trained index or stored codebook.
    h = np.array(hidden[:, :32], dtype=np.float32, order="C", copy=True)
    h /= np.maximum(1e-12, np.linalg.norm(h, axis=1, keepdims=True))
    observed = p[np.arange(len(y)), y]
    original = np.maximum(3e-8, observed)
    deltas = np.zeros(len(options))
    for start in range(BLOCK, len(y), BLOCK):
        stop = min(start + BLOCK, len(y))
        first = max(0, start - 512)
        # All four neighbours must precede the current block, including its first query.
        similarities = h[start:stop] @ h[first:start].T
        rows = np.arange(stop - start)
        indices = np.empty((len(rows), 4), dtype=int)
        cosines = np.empty_like(indices, dtype=np.float64)
        for k in range(4):
            index = similarities.argmax(axis=1)
            indices[:, k] = index
            cosines[:, k] = similarities[rows, index]
            similarities[rows, index] = -2
        weights = np.exp(12 * (cosines - cosines[:, :1]))
        correct = y[first + indices] == y[start:stop, None]
        q = (correct * weights).sum(1) / weights.sum(1)
        for i, (minimum, weight) in enumerate(options):
            mix = weight * (cosines[:, 0] >= minimum)
            revised = (1 - mix) * observed[start:stop] + mix * q
            deltas[i] += np.log2(original[start:stop] / np.maximum(3e-8, revised)).sum()
    return dict(document=window["document"], targets=len(y), delta_bits=deltas.tolist())


def run(args):
    began = time.monotonic()
    if file_sha(args.model) != REFERENCE_SHA:
        raise ValueError("expected the published model")
    suite = args.data / "training"
    windows = json.loads((suite / "manifest.json").read_bytes())["windows"]
    if len(windows) != 1024 or len({w["document"] for w in windows}) != 1024:
        raise ValueError("expected 1024 distinct training articles")
    selection = windows[4::8]
    other = [w for i, w in enumerate(windows) if i % 8 != 4]
    cache = ProfileCache(args.cache, args.exe, args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.experiment == "frequency":
        evaluator, options, fields = score, OPTIONS, ("prior", "strength")
    else:
        evaluator, options, fields = retrieve, RETRIEVAL, ("cosine", "weight")
    with article_pool(args.cores) as pool:
        articles = list(pool.map(partial(evaluator, cache, suite, options), selection))
        deltas = np.sum([w["delta_bits"] for w in articles], axis=0)
        n = sum(w["targets"] for w in articles)
        winner = int(np.argmin(deltas))
        frozen = dict(zip(fields, options[winner]))
        frozen.update(
            experiment=args.experiment,
            block=BLOCK,
            selection_articles=len(selection),
            selection_targets=n,
            selection_bits_per_target_by_option=(deltas / n).tolist(),
            options=options,
        )
        (args.output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
        fn = partial(evaluator, cache, suite, [options[winner]])
        confirm = list(pool.map(fn, other))
    n = sum(w["targets"] for w in confirm)
    bits = sum(w["delta_bits"][0] for w in confirm)
    lookup = {w["document"]: w["stream_byte"] for w in other}
    ordered = sorted(confirm, key=lambda row: lookup[row["document"]])
    values = np.array([(row["delta_bits"][0], row["targets"]) for row in ordered])
    rng = np.random.default_rng(23)
    draws = np.zeros((2000, 2))
    for band in np.array_split(values, 8):
        draws += band[rng.integers(len(band), size=(len(draws), len(band)))].sum(axis=1)
    summary = dict(
        frozen=frozen,
        confirmation_articles=len(confirm),
        confirmation_targets=n,
        confirmation_bits_per_target=bits / n,
        article_bootstrap_95=np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975]).tolist(),
        projected_primary_head_bytes=bits / n * FULL_TARGETS / 8,
        improved_articles=sum(w["delta_bits"][0] < 0 for w in confirm),
        elapsed_seconds=time.monotonic() - began,
    )
    (args.output / "result.json").write_text(
        json.dumps(dict(summary, articles=confirm), indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "exe", "model", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--cores", required=True, nargs="+", type=int)
    parser.add_argument("--experiment", choices=("frequency", "retrieval"), default="frequency")
    run(parser.parse_args())
