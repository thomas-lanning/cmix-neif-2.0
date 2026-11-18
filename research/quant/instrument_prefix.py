"""Instrument a scratch release tree to measure a true mixed-coder prefix."""
from pathlib import Path
import sys


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"release source changed at {old[:60]}")
    return source.replace(old, new, 1)


root = Path(sys.argv[1])
capture_bits = "--no-events" not in sys.argv[2:]
runner = root / "src/runner.cpp"
src = runner.read_text()
# The caller supplies the bit-identical published dictionary and article order.
# Keep the entire corpus preprocessing path and final predictor as shipped.
start = src.index("    if (selfextract_comp() != 0)")
end = src.index('    console_progress::Phase("preprocessing");', start)
src = src[:start] + src[end:]
src = once(src, '    console_progress::Phase("preprocessing");', '''    if (const char* canonical = std::getenv("NEIF_CANONICAL_STREAM")) {
      FILE* source = fopen(canonical, "rb");
      FILE* output = fopen(output_path.c_str(), "wb");
      if (!source || !output || fseek(source, 0, SEEK_END)) std::exit(7);
      const long length = ftell(source);
      if (length != 501062281 || fseek(source, 0, SEEK_SET)) std::exit(7);
      census_block_active = true;
      isbn_check::after_at.an = true;
      std::vector<bool> vocab(256, false);
      vocab_frozen::FillVocab(&vocab);
      WriteHeaderVocfix(length, true, output);
      dict_article::strom = length;
      dict_article::wrt = true;
      dict_article::Verbergen();
      Predictor predictor(vocab);
      unsigned long long bytes = 0;
      Compress(length, source, output, &bytes, &predictor);
      std::exit(9);
    }
    console_progress::Phase("preprocessing");''')
src = once(src, "  Encoder e(os, p);", """  Encoder e(os, p);
  FILE* log = fopen("prefix-sizes.tsv", "wx");
  if (!log) std::exit(2);
  bool capture = std::getenv("NEIF_CAPTURE_ARTICLE") != nullptr;
  bool initial = std::getenv("NEIF_CAPTURE_INITIAL") != nullptr;
  unsigned long long stop = std::getenv("NEIF_QUICK_CHECK") ? 200000 : 20000000;""")
src = once(src, "      p->SemByteStart(sem_pb);", """      research_capture = (capture && pos >= 16000000 && pos < 16262144) || (initial && pos < 20000);
      p->SemByteStart(sem_pb);""")
src = once(src, "      ++pos;\n    }\n  }\n  e.Flush();", """      ++pos;
      if (pos == 15282108 || pos == 16000000 || pos == 17000000 || pos == 18000000 || pos == 20000000) {
        fprintf(log, "%llu\\t%zu\\n", pos, e.OutputSize());
        fflush(log);
      }
      if (pos == stop) {
        e.Flush();
        if (fclose(log) || fflush(os)) std::exit(4);
        std::exit(0);
      }
    }
  }
  e.Flush();""")
runner.write_text(("extern bool research_capture;\n" if capture_bits else "static bool research_capture;\n") + src)

encoder = root / "src/coder/encoder.cpp"
src = encoder.read_text()
src = once(src, "  const unsigned int xmid = floor_stage::Mid(x1_, x2_, p_->SemCorrect(",
           "  const unsigned int actual_p = p_->SemCorrect(")
src = once(src, "(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7)));",
           "(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7));\n"
           "  if (research_capture) research_event(actual_p, bit);\n"
           "  const unsigned int xmid = floor_stage::Mid(x1_, x2_, actual_p);")
src = once(src, "  if (p > 65534) p = 65534;", "  if (p > 65534) p = 65534;\n"
           "  if (research_capture) research_event((uint32_t)p << 16, bit);")
if capture_bits:
    encoder.write_text("""#include <cstdio>
#include <cstdint>
#include <cstdlib>
bool research_capture = false;
static void research_event(uint32_t probability, int bit) {
  static FILE* out = std::fopen("article-events.bin", "wb");
  unsigned char symbol = bit;
  if (!out || std::fwrite(&probability, 4, 1, out) != 1 || std::fwrite(&symbol, 1, 1, out) != 1)
    std::exit(3);
}
""" + src)

reader = root / "src/tf1/weights_io_compressed.cpp"
src = once(reader.read_text(), "  if (h != 0xaec368ac6a568632ULL)",
    """  const char* expected = std::getenv("NEIF_CANDIDATE_FNV");
  if (!expected || !*expected || size < 24 || n != size || std::memcmp(buf, "FX2TFWP5", 8) ||
      h != std::strtoull(expected, nullptr, 16))""")
reader.write_text(src)
