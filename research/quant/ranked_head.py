"""Select the width of a small byte predictor by predicted payload plus stored bytes."""
import argparse
from functools import partial
import json
from pathlib import Path
import struct

import numpy as np

from context_screen import read_article
from evaluate import article_pool, FULL_TARGETS, REFERENCE_SHA
from profile_cache import file_sha, ProfileCache

FEATURES = 64
VOCAB = 208
RANKS = (1, 4, 16, 32)
GAINS = (0.125, 0.5, 1.0)


def article(cache, suite, w):
    p, y, hidden, _ = read_article(cache, suite, w)
    x = np.where(hidden[:, :FEATURES] > 0, 1.0, -1.0)
    return x, p, y


def moments(cache, suite, w):
    x, p, y = article(cache, suite, w)
    variance = (p * (1 - p)).sum(0)
    residual = -p
    residual[np.arange(len(y)), y] += 1
    return x.T @ x, x.T @ residual, variance, len(y)


def encode_factors(left, right, gain):
    # One byte for rank, then two float scales and signed byte factors per component.
    data = bytearray(struct.pack("<B", len(left)))
    for a, b in zip(left, right, strict=True):
        sa = max(np.abs(a).max() / 127, 1e-20)
        sb = max(np.abs(b).max() / 127, 1e-20)
        data += struct.pack("<ff", sa, sb * gain)
        data += np.rint(a / sa).clip(-127, 127).astype("i1").tobytes()
        data += np.rint(b / sb).clip(-127, 127).astype("i1").tobytes()
    return bytes(data)


def decode_factors(blob):
    if not blob:
        return np.zeros((FEATURES, VOCAB))
    rank = blob[0]
    stride = 8 + FEATURES + VOCAB
    if len(blob) != 1 + rank * stride:
        raise ValueError("invalid factor file length")
    w = np.zeros((FEATURES, VOCAB))
    for k in range(rank):
        pos = 1 + k * stride
        sa, sb = struct.unpack_from("<ff", blob, pos)
        a = np.frombuffer(blob, "i1", FEATURES, pos + 8).astype(float)
        b = np.frombuffer(blob, "i1", VOCAB, pos + 8 + FEATURES).astype(float)
        w += np.outer(a * sa, b * sb)
    return w


def fit(cov, gradient, variance, n):
    # One regularized second-order step using feature covariance and diagonal class curvature.
    cov = cov / n + 0.05 * np.eye(FEATURES)
    lower = np.linalg.cholesky(cov)
    scale = np.sqrt(np.maximum(variance / n, 0.00025))
    normalized = np.linalg.solve(lower, gradient / n) / scale
    u, s, vh = np.linalg.svd(normalized, full_matrices=False)
    left = np.linalg.solve(lower.T, u).T
    right = s[:, None] * vh / scale
    models = [(0, 0.0, b"")]
    for rank in RANKS:
        for gain in GAINS:
            models.append((rank, gain, encode_factors(left[:rank], right[:rank], gain)))
    return models


def losses(cache, suite, matrices, w):
    x, p, y = article(cache, suite, w)
    original = np.maximum(3e-8, p[np.arange(len(y)), y])
    totals = []
    for matrix in matrices:
        if matrix is None:
            totals.append(0.0)
            continue
        multiplier = np.exp(np.clip(x @ matrix, -2.0, 2.0))
        revised = (p[np.arange(len(y)), y] * multiplier[np.arange(len(y)), y]) / (
            p * multiplier
        ).sum(1)
        totals.append(float(np.log2(original / np.maximum(3e-8, revised)).sum()))
    return len(y), totals


def measure(pool, cache, suite, windows, models):
    matrices = [decode_factors(blob) if blob else None for _, _, blob in models]
    result = list(pool.map(partial(losses, cache, suite, matrices), windows))
    bits = np.array([d for _, d in result])
    targets = np.array([n for n, _ in result])
    return bits, targets


def run(args):
    if file_sha(args.model) != REFERENCE_SHA:
        raise ValueError("expected released baseline")
    windows = json.loads((args.suite / "manifest.json").read_bytes())["windows"]
    if len(windows) != 1024 or len({w["document"] for w in windows}) != len(windows):
        raise ValueError("expected 1024 distinct articles")
    train = [w for i, w in enumerate(windows) if i % 8 < 6]
    select = windows[6::8]
    confirm = windows[7::8]
    args.output.mkdir(parents=True, exist_ok=False)
    cache = ProfileCache(args.cache, args.exe, args.model)
    with article_pool(args.cores) as pool:
        cov = np.zeros((FEATURES, FEATURES))
        grad = np.zeros((FEATURES, VOCAB))
        variance = np.zeros(VOCAB)
        n = 0
        for c, g, v, t in pool.map(partial(moments, cache, args.suite), train):
            cov += c
            grad += g
            variance += v
            n += t
        models = fit(cov, grad, variance, n)
        bits, targets = measure(pool, cache, args.suite, select, models)
        cost = np.array([2 * len(blob) for _, _, blob in models])
        projected = args.transfer * FULL_TARGETS / 8 * bits.sum(0) / targets.sum() + cost
        winner = int(np.argmin(projected))
        rank, gain, blob = models[winner]
        table = args.output / "frozen.bin"
        table.write_bytes(blob)
        record = dict(
            rank=rank,
            gain=gain,
            bytes_per_copy=len(blob),
            sha256=file_sha(table),
            fit_articles=len(train),
            selection_articles=len(select),
            fit_targets=n,
            selection_targets=int(targets.sum()),
            transfer=args.transfer,
            candidates=[
                dict(rank=r, gain=g, bytes_per_copy=len(b), estimated_entry_delta=float(d))
                for (r, g, b), d in zip(models, projected, strict=True)
            ],
        )
        (args.output / "frozen.json").write_text(json.dumps(record, indent=2) + "\n")
        bits, targets = measure(
            pool, cache, args.suite, confirm, [(rank, gain, table.read_bytes())]
        )
    data = np.column_stack((bits[:, 0], targets))
    order = np.argsort([w["stream_byte"] for w in confirm])
    rng = np.random.default_rng(2301)
    draws = np.zeros((2000, 2))
    for band in np.array_split(data[order], 8):
        draws += band[rng.integers(len(band), size=(len(draws), len(band)))].sum(1)
    delta = bits.sum() / targets.sum() * FULL_TARGETS / 8
    result = dict(
        frozen=record,
        confirmation_articles=len(confirm),
        confirmation_targets=int(targets.sum()),
        primary_head_byte_delta=float(delta),
        estimated_entry_delta=float(2 * len(blob) + args.transfer * delta),
        article_bootstrap_95=(
            2 * len(blob)
            + args.transfer
            * FULL_TARGETS
            / 8
            * np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975])
        ).tolist(),
        improved_articles=int((bits[:, 0] < 0).sum()),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("suite", "exe", "model", "cache", "output"):
        p.add_argument(field, type=Path)
    p.add_argument("--cores", required=True, nargs="+", type=int)
    p.add_argument("--transfer", required=True, type=float)
    args = p.parse_args()
    if not np.isfinite(args.transfer) or args.transfer <= 0:
        p.error("transfer must be positive")
    run(args)
