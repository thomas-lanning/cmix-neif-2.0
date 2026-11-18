// SPDX-License-Identifier: GPL-3.0-only
// Measure causal recalibration on the final probabilities recorded by the release.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Event { uint32_t p; uint8_t bit; };
constexpr uint64_t one = uint64_t{1} << 32;

class Recalibration {
  struct Cell { uint64_t expected = 0; uint32_t wrong = 0, count = 0; };
  std::array<Cell, 2048> cells{};
  unsigned history = 0, index = 0;
  uint32_t error = 0;
  bool side = false;
  unsigned prior, previous, strength, half;
public:
  Recalibration(unsigned k, unsigned h, unsigned s, unsigned n)
      : prior(k), previous(h), strength(s), half(n) {}

  uint32_t predict(uint32_t p) {
    side = p >= one / 2;
    error = side ? one - p : p;
    // Four buckets per binary exponent. Upper bits of the confidence determine
    // the bucket without a logarithm. History uses only previous coded bits.
    unsigned exponent = 31u - __builtin_clz(error);
    unsigned bucket = exponent * 4 + ((uint64_t(error) << 2 >> exponent) & 3u);
    index = ((bucket * 2 + side) << previous) + (history & ((1u << previous) - 1));
    const Cell& c = cells[index];
    uint64_t numerator = uint64_t(c.wrong + prior) << 16;
    uint64_t denominator = c.expected + (uint64_t(prior) << 16);
    // Bound the ratio before applying it. Both products fit unsigned 64 bits
    // by the count limit and Q16 expected-error representation.
    uint64_t gain = std::clamp((numerator << 12) / denominator, uint64_t{2048}, uint64_t{8192});
    gain = (gain * strength + 4096 * (4 - strength)) / 4;
    uint64_t corrected = std::clamp((uint64_t(error)*gain + 2048) >> 12, uint64_t{1}, one-1);
    return side ? one - corrected : corrected;
  }

  void observe(unsigned bit) {
    Cell& c = cells[index];
    c.expected += (uint64_t(error) + 32768) >> 16;
    c.wrong += bit != side;
    if (++c.count >= half) { c.count >>= 1; c.wrong >>= 1; c.expected >>= 1; }
    history = history * 2 + bit;
  }
};

class Arithmetic {
  uint32_t lo = 0, hi = UINT32_MAX;
public:
  uint64_t bytes = 0;
  void apply(uint32_t p, unsigned bit) {
    uint32_t mid = lo + ((uint64_t(hi-lo)*p) >> 32);
    if (bit) hi=mid; else lo=mid+1;
    while ((lo >> 24) == (hi >> 24)) { ++bytes; lo <<= 8; hi=(hi << 8) | 255; }
  }
};

int main(int argc, char** argv) {
  if (argc!=2) return 2;
  std::ifstream input(argv[1],std::ios::binary);
  if (!input) return 2;
  std::vector<Event> events;
  std::array<uint8_t,5> buf;
  while (input.read(reinterpret_cast<char*>(buf.data()), buf.size())) {
    uint32_t p=buf[0] | (uint32_t(buf[1])<<8) | (uint32_t(buf[2])<<16) | (uint32_t(buf[3])<<24);
    if (!p || buf[4]>1) return 3;
    events.push_back({p,buf[4]});
  }
  if (!input.eof() || input.gcount()!=0 || events.size()<4) return 3;
  for (unsigned prior : {4u,16u,64u}) for (unsigned previous : {0u,1u,3u})
    for (unsigned strength : {1u,2u,4u}) for (unsigned half : {4096u,65536u}) {
      Recalibration adjust(prior,previous,strength,half);
      Arithmetic base,candidate;
      std::array<int64_t,4> deltas{};
      for (size_t i=0; i<events.size(); ++i) {
        const Event& e=events[i];
        base.apply(e.p,e.bit);
        candidate.apply(adjust.predict(e.p),e.bit);
        adjust.observe(e.bit);
        for (unsigned q=0; q<4; ++q) if (i+1==events.size()*(q+1)/4)
          deltas[q]=int64_t(candidate.bytes)-int64_t(base.bytes);
      }
      // Each stream needs one byte to terminate. The difference cancels.
      std::printf("{\"prior\":%u,\"history\":%u,\"strength_quarters\":%u,\"half\":%u,"
                  "\"events\":%zu,\"arithmetic_bytes\":%llu,\"quarter_deltas\":[%lld,%lld,%lld,%lld]}\n",
                  prior,previous,strength,half,events.size(),(unsigned long long)base.bytes+1,
                  (long long)deltas[0],(long long)(deltas[1]-deltas[0]),(long long)(deltas[2]-deltas[1]),
                  (long long)(deltas[3]-deltas[2]));
    }
}
