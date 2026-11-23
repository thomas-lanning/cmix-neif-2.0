"""Install a frozen rank-one head into a fresh research source tree."""
import argparse
import hashlib
from pathlib import Path
import struct


def header(blob):
    if len(blob) != 281 or blob[0] != 1:
        raise ValueError("expected one 64 by 208 factored correction")
    sa, sb = struct.unpack_from("<ff", blob, 1)
    left = struct.unpack_from("<64b", blob, 9)
    right = struct.unpack_from("<208b", blob, 73)
    integers = lambda x: ", ".join(map(str, x))
    bound = sum(map(abs, left))
    return f"""// Generated from the frozen native-head fit, SHA-256 {hashlib.sha256(blob).hexdigest()}
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ranked_head {{
static constexpr int bound = {bound};
static constexpr int8_t left[64] = {{{integers(left)}}};
static constexpr int8_t right[208] = {{{integers(right)}}};
struct Factors {{
    double values[2 * bound + 1][208];
    Factors() {{
        for (int d = -bound; d <= bound; ++d) {{
            for (int j = 0; j < 208; ++j) {{
                const double x = d * {sa.hex()} * {sb.hex()} * right[j];
                values[d + bound][j] = std::exp(std::max(-2.0, std::min(2.0, x)));
            }}
        }}
    }}
}};
inline void correct(const float* hidden, float* probabilities) {{
    static const Factors factors;
    int dot = 0;
    for (int i = 0; i < 64; ++i) dot += hidden[i] > 0 ? left[i] : -left[i];
    const double* row = factors.values[dot + bound];
    double adjusted[208], sum = 0;
    for (int j = 0; j < 208; ++j) {{
        adjusted[j] = probabilities[j] * row[j];
        sum += adjusted[j];
    }}
    for (int j = 0; j < 208; ++j) probabilities[j] = float(adjusted[j] / sum);
}}
}}
"""


def install(tree, model, expected):
    blob = model.read_bytes()
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("frozen factors differ")
    folder = tree / "src/tf1/opt"
    source = folder / "model_opt.cpp"
    text = source.read_text()
    anchor = "      head_plm_softcap_softmax(logits, prior_f32, dot192(M.plm_w, xn), probs208);"
    if text.count(anchor) != 1:
        raise ValueError("the primary output path differs")
    with (folder / "rank_one.h").open("x") as out:
        out.write(header(blob))
    source.write_text(
        '#include "rank_one.h"\n'
        + text.replace(anchor, anchor + "\n      ranked_head::correct(xn, probs208);")
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tree", type=Path)
    p.add_argument("factors", type=Path)
    p.add_argument("sha256")
    a = p.parse_args()
    install(a.tree, a.factors, a.sha256)
