"""Prepare an instrumented research build without editing the release sources."""
from pathlib import Path
import struct
import sys
import hashlib

root = Path(sys.argv[2])
if sys.argv[1] == "instrument":
    p = root / "src/coder/decoder.cpp"
    src = p.read_text()
    a = "const unsigned int xmid = floor_stage::Mid(x1_, x2_, p_->SemCorrect("
    b = "(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7)));"
    if src.count(a) != 1 or src.count(b) != 1 or src.count("  return bit;\n}") != 2:
        raise ValueError("decoder changed; review the capture points")
    src = src.replace(a, "const uint32_t model_p = p_->SemCorrect(").replace(
        b, "(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7));\n  const unsigned xmid = floor_stage::Mid(x1_, x2_, model_p);")
    src = src.replace("  return bit;\n}", "  record_event(model_p, bit);\n  return bit;\n}", 1)
    idx = src.index("int Decoder::DecodeRawBit(")
    src = src[:idx] + src[idx:].replace("  return bit;\n}", "  record_event(uint32_t(p) << 16, bit);\n  return bit;\n}", 1)
    header = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
static void record_event(uint32_t probability, int bit) {
  static FILE* out = std::fopen("coded-events.bin", "wb");
  static unsigned n = 0;
  unsigned char actual = bit;
  if (!out || std::fwrite(&probability, 4, 1, out) != 1 || std::fwrite(&actual, 1, 1, out) != 1)
    std::_Exit(2);
  if (++n == 131072) std::_Exit(std::fclose(out) == 0 ? 0 : 3);
}
"""
    p.write_text(header + src)
elif sys.argv[1] == "payload":
    archive = root.read_bytes()
    if hashlib.sha256(archive).hexdigest() != "88dd7087404122d010261e6855afd88875d34dde1f0539377d3e832fc0f6b3fd":
        raise ValueError("invalid upstream archive")
    length, weights = struct.unpack_from("<II", archive, len(archive)-8)
    if weights != 3744629 or not 0 < length < len(archive)-8-weights:
        raise ValueError("invalid archive trailer")
    Path("published.payload").write_bytes(archive[-8-length:-8])
else:
    raise SystemExit("expected instrument or payload")
