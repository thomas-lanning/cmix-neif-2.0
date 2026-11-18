// SPDX-License-Identifier: GPL-3.0-only
// Streaming binary rANS for cmix. Both sides count actual entropy decisions.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace model_rans {
constexpr size_t block_size = 1u << 18;
constexpr uint64_t lower = uint64_t{1} << 40;
constexpr uint64_t total = uint64_t{1} << 32;

inline void require(bool ok) {
  if (!ok) {
    std::fputs("Invalid or incomplete rANS stream\n", stderr);
    std::exit(6);
  }
}

class Encoder {
  std::vector<uint32_t> probabilities;
  std::vector<uint8_t> outcomes, reversed;
  size_t written = 0;
public:
  void push(FILE* file, uint32_t probability, int bit) {
    require(probability && (bit == 0 || bit == 1));
    probabilities.push_back(probability);
    outcomes.push_back(uint8_t(bit));
    if (outcomes.size() == block_size) flush(file);
  }
  void flush(FILE* file) {
    if (outcomes.empty()) return;
    uint64_t state = lower;
    reversed.clear();
    for (size_t i = outcomes.size(); i--;) {
      uint64_t frequency = probabilities[i];
      uint64_t start = outcomes[i] ? 0 : frequency;
      uint64_t width = outcomes[i] ? frequency : total - frequency;
      while (state >= ((lower >> 32) << 8) * width) {
        reversed.push_back(uint8_t(state));
        state >>= 8;
      }
      state = ((state / width) << 32) + state % width + start;
    }
    for (unsigned i = 0; i < 6; ++i)
      require(std::putc(int((state >> (8 * i)) & 255), file) != EOF);
    std::reverse(reversed.begin(), reversed.end());
    if (!reversed.empty())
      require(std::fwrite(reversed.data(), 1, reversed.size(), file) == reversed.size());
    written += 6 + reversed.size();
    probabilities.clear();
    outcomes.clear();
  }
  size_t size() const { return written; }
};

class Decoder {
  uint64_t state = lower;
  size_t used = 0;
  static unsigned read_byte(FILE* file) {
    int c = std::getc(file);
    require(c != EOF);
    return unsigned(c);
  }
public:
  int next(FILE* file, uint32_t probability) {
    require(probability != 0);
    if (used == 0) {
      require(state == lower);
      state = 0;
      for (unsigned i = 0; i < 6; ++i)
        state |= uint64_t(read_byte(file)) << (8 * i);
      require(state >= lower && state < (lower << 8));
    }
    uint64_t slot = uint32_t(state);
    int bit = slot < probability;
    uint64_t width = bit ? probability : total - probability;
    state = width * (state >> 32) + slot - (bit ? 0 : probability);
    while (state < lower) state = (state << 8) | read_byte(file);
    if (++used == block_size) used = 0;
    return bit;
  }
  void finish(FILE* file) const {
    require(state == lower && std::getc(file) == EOF && !std::ferror(file));
  }
};
}
