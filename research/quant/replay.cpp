// SPDX-License-Identifier: GPL-3.0-only
// Quantization screen for the primary transformer head using exported NP2 weights.
#include "../../src/tf1/opt/model_opt.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fx2 { unsigned long long research_fingerprint; }

struct Row {
  uint8_t token, prev1, prev2, match, match_class, target, score, reset;
  uint16_t prior[208];
};
static_assert(sizeof(Row) == 424, "fixture row layout");

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  char* end = nullptr;
  fx2::research_fingerprint = std::strtoull(argv[2], &end, 16);
  if (!*argv[2] || *end) return 2;
  FILE* file = std::fopen(argv[3], "rb");
  char magic[8];
  if (!file || std::fread(magic, 1, 8, file) != 8 || std::memcmp(magic, "NEIFQ001", 8)) return 2;
  fx2::opt::TransformerOpt model(argv[1], fx2::opt::AttnKind::KVI8);
  alignas(64) std::array<float, 224> probs{};
  Row row{};
  unsigned rows = 0, targets = 0;
  double bits = 0;
  for (;;) {
    size_t n = std::fread(&row, 1, sizeof(row), file);
    if (!n && !std::ferror(file)) break;
    if (n != sizeof(row) || row.token >= 216 || row.prev1 > 216 || row.prev2 > 216 ||
        row.match > 208 || row.match_class > 15 || row.target >= 208 || row.score > 1 ||
        row.reset > 1 || (!rows && !row.reset)) return 3;
    if (row.reset) model.begin_article();
    fx2::PaketChannels channels{row.prev1, row.prev2, row.match, row.match_class};
    model.set_gelesen(row.score);
    model.step(row.token, row.prior, probs.data(), channels);
    if (row.score) {
      double p = probs[row.target];
      if (!std::isfinite(p) || p < 0 || p > 1.00001) return 4;
      bits -= std::log2(std::max(p, 3e-8));
      ++targets;
    }
    ++rows;
  }
  if (!targets) return 5;
  std::printf("{\"rows\":%u,\"targets\":%u,\"bits\":%.12f}\n", rows, targets, bits);
}
