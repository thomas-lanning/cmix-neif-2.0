"""Install the experimental rANS coder into a fresh copy of the pinned cmix sources."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"unexpected upstream source at {old[:65]!r}")
    return text.replace(old, new)


def between(text, first, end, new):
    lo = text.index(first)
    hi = text.index(end, lo)
    return text[:lo] + new + text[hi:]


def install(root, prefix, continuous=False):
    if prefix:
        subprocess.run([sys.executable, HERE.parent/"quant/instrument_prefix.py", root, "--no-events"], check=True)
    coder = root/"src/coder"
    shutil.copyfile(HERE/("continuous_rans.h" if continuous else "model_rans.h"), coder/"model_rans.h")
    for mode in ("encoder", "decoder"):
        file = coder/(mode+".h")
        text = replace(file.read_text(), '#include <stdio.h>', '#include <stdio.h>\n#include "model_rans.h"')
        text = replace(text, " private:", f" private:\n  model_rans::{mode.title()} rans_;")
        if mode == "encoder":
            text = replace(text, "return total_out_ + out_n_;", "return rans_.size();")
        else:
            text = replace(text, "  int Decode();", "  int Decode();\n  void Finish() { rans_.finish(is_); }")
        file.write_text(text)

    for mode in ("encoder", "decoder"):
        file = coder/(mode+".cpp")
        text = replace(file.read_text(), "const unsigned int xmid = floor_stage::Mid(x1_, x2_, p_->SemCorrect(",
                       "const unsigned int probability = p_->SemCorrect(")
        text = replace(text, "(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7)));",
                       "(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7));")
        if mode == "encoder":
            text = between(text, "  if (bit) {", "  floor_.Update(bit);", "  rans_.push(os_, probability, bit);\n")
            text = between(text, "  while (((x1_^x2_)", "void Encoder::EncodeRawBit", "}\n\n")
            text = between(text, "  const unsigned int xmid =", "void Encoder::ObserveKnownByte",
                           "  rans_.push(os_, uint32_t(p) << 16, bit);\n}\n\n")
            text = text[:text.index("__attribute__((cold)) void Encoder::Flush()")] + (
                "__attribute__((cold)) void Encoder::Flush() { rans_.flush(os_); }\n")
        else:
            text = between(text, "  for (int i = 0; i < 4; ++i)", "// Past the end", "}\n\n")
            text = between(text, "  int bit = 0;", "  floor_.Update(bit);", "  int bit = rans_.next(is_, probability);\n")
            text = between(text, "  while (((x1_^x2_)", "  return bit;", "")
            text = between(text, "  const unsigned int xmid =", "void Decoder::ObserveKnownByte",
                           "  return rans_.next(is_, uint32_t(p) << 16);\n}\n\n")
        file.write_text(text)

    runner = root/"src/runner.cpp"
    text = replace(runner.read_text(), '  run_mode::gram.Report("decompress");',
                   '  d.Finish();\n  run_mode::gram.Report("decompress");')
    if prefix:
        # Shadow arithmetic in this research build independently validates all
        # model decisions against the published continuing payload.
        file = coder/"encoder.cpp"
        shadow = """
#include <cstdio>
#include <cstdlib>
#include <cstdint>
static void reference_event(uint32_t probability, int bit) {
  static FILE* ref = std::fopen("arithmetic.oracle", "wb");
  static uint32_t lo = 0, hi = UINT32_MAX;
  if (!ref) std::exit(6);
  const uint32_t mid = lo + ((uint64_t(hi-lo)*probability) >> 32);
  if (bit) hi=mid; else lo=mid+1;
  while ((lo >> 24) == (hi >> 24)) {
    if (std::putc(hi >> 24, ref) == EOF) std::exit(6);
    lo <<= 8; hi=(hi << 8) | 255;
  }
}
"""
        code = file.read_text()
        for variable in ("probability", "uint32_t(p) << 16"):
            call = f"rans_.push(os_, {variable}, bit);"
            code = replace(code, call, f"reference_event({variable}, bit);\n  "+call)
        file.write_text(shadow+code)
        anchor = "  if (argc == 1) {\n    //Decompress enwik9"
        text = replace(text, anchor, """
  if (argc == 4 && std::getenv("NEIF_RANS_REPLAY") && argv[1][1] == 'd') {
    FILE* input = fopen(input_path.c_str(), "rb");
    FILE* output = fopen(output_path.c_str(), "wb");
    if (!input || !output) return 7;
    unsigned long long length;
    bool dictionary_used;
    std::vector<bool> vocab(256, false);
    ReadHeaderVocfix(input, &length, &dictionary_used, &vocab);
    if (length != 501062281 || !dictionary_used) return 7;
    census_block_active = true;
    isbn_check::after_at.an = true;
    dict_article::strom = length;
    dict_article::wrt = true;
    dict_article::Verbergen();
    Predictor predictor(vocab);
    Decompress(length, input, output, &predictor);
    return fclose(output) || fclose(input);
  }
""" + anchor)
        anchor = "    sem_pb = byte & 255;   // the stream byte"
        text = replace(text, anchor, """    if (pos + 1 == (std::getenv("NEIF_QUICK_CHECK") ? 200000u : 20000000u)) {
      d.Finish();
      return;
    }
""" + anchor)
    runner.write_text(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--prefix-test", action="store_true", help="enable bounded canonical encode, decode and reference trace")
    p.add_argument("--continuous", action="store_true", help="use one 64-bit state and a bounded-RAM disk spool")
    args = p.parse_args()
    install(args.source.resolve(), args.prefix_test, args.continuous)
