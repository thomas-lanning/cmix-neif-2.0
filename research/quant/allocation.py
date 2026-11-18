"""Post-training experiments on the shipped, fixed-geometry NP2 tensors."""
import math
import re
import numpy as np

from candidate import np2


def bf16(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def round_bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def reduce_rows(q, scales, fraction, limit):
    """Fit a smaller signed alphabet and a representable scale to each row."""
    raw = q.astype(np.float64)
    peak = np.max(np.abs(raw), axis=1)
    eligible = np.flatnonzero(peak > limit)
    if not len(eligible):
        return 0
    x = raw[eligible]
    gain = peak[eligible] / limit
    for _ in range(4):
        codes = np.clip(np.rint(x / gain[:, None]), -limit, limit)
        gain = (x * codes).sum(axis=1) / np.maximum((codes * codes).sum(axis=1), 1)
    scale = bf16(scales[eligible]).astype(np.float64)
    fitted = round_bf16(scale * gain)
    physical = bf16(fitted).astype(np.float64)
    if not np.isfinite(physical).all() or np.any(physical == 0) or np.any(scale == 0):
        raise ValueError("invalid row scale")
    codes = np.clip(np.rint(x * (scale / physical)[:, None]), -limit, limit)
    error = np.mean((x * scale[:, None] - codes * physical[:, None]) ** 2, axis=1)
    count = max(1, math.ceil(len(eligible) * fraction))
    selected = np.argsort(error, kind="stable")[:count]
    q[eligible[selected]] = codes[selected].astype(np.int8)
    scales[eligible[selected]] = fitted[selected]
    return count


def apply(tensors, spec):
    names = {t.name: t for t in tensors}
    changed = 0
    for t in tensors:
        match = re.fullmatch(r"blocks\.(\d+)\.(.+)\.weight\.q", t.name)
        if not match or int(match[1]) not in spec["layers"]:
            continue
        site = match[2]
        family = spec["family"]
        if family == "attention":
            if not site.startswith("attention."):
                continue
        elif site != "mlp." + family:
            continue
        # Only paired per-output BF16 scales belong to this experiment.
        scale_t = names.get(t.name[:-1] + "scale")
        if scale_t is None or scale_t.a.dtype != np.uint16 or scale_t.a.size != t.rows():
            continue
        scales = scale_t.a.reshape(-1)
        if spec["kind"] == "scale":
            drop = 7 - spec["mantissa"]
            wide = scales.astype(np.uint32)
            quant = ((wide + (1 << (drop-1))-1 + ((wide >> drop) & 1)) >> drop) << drop
            changed += int(np.count_nonzero(scales != quant))
            scales[:] = quant.astype(np.uint16)
        else:
            changed += reduce_rows(t.m2(), scales, spec["fraction"], spec["levels"])
    if not changed:
        raise ValueError(f"no eligible changes for {spec}")
    return changed


def export(baseline, specs, output):
    tensors = [np2.Ent(t.name, t.code, t.a) for t in baseline]
    counts = [apply(tensors, s) for s in specs]
    blob, _, permuted = np2.encode_np(tensors, True, np2.np_groups(tensors, True, True), True)
    with output.open("xb") as f:
        f.write(blob)
    actual = np2.read_container(str(output))
    if len(actual) != len(permuted) or not all(a.same(b) for a, b in zip(actual, permuted)):
        raise ValueError("export differs from intended permuted tensors")
    return counts
