// SPDX-License-Identifier: GPL-3.0-only
// Score possible int4 representations of a semantic-cache key by cosine.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace research_cache {
constexpr int width = 128;
// A numerator of 14 is the published mapping to [-7,7]. The other mappings
// can use the sixteenth signed nibble (-8). Half-even ties are deterministic.
constexpr int numerators[] = {14,15,16,18,20};
struct Tables {
  int8_t codes[5][255]{};
  Tables() {
    // The projecting kernel maps the largest absolute component to exactly
    // 127. A zero input stays zero. Every key therefore shares these tables.
    for (int i = 0; i < 5; ++i)
      for (int x = -127; x <= 127; ++x) {
        int num = std::abs(x)*numerators[i], den = 254;
        int v = num/den, rem = num%den;
        if (2*rem > den || (2*rem == den && (v&1))) ++v;
        codes[i][x+127] = int8_t(std::clamp(x<0?-v:v, -8, 7));
      }
  }
};

inline void pack(uint8_t* output, const int8_t* key, bool optimize) {
  static const Tables tables;
  int best = 2;
  if (optimize) {
    int64_t best_squared = -1, best_norm = 1;
    for (int i = 0; i < 5; ++i) {
      int64_t dot = 0, norm = 0;
      const int8_t* values = tables.codes[i]+127;
      for (int j = 0; j < width; ++j) {
        int v=values[key[j]];
        dot+=key[j]*v; norm+=v*v;
      }
      if (dot*dot*best_norm > best_squared*norm) {
        best_squared=dot*dot; best_norm=norm; best=i;
      }
    }
  }
  const int8_t* values=tables.codes[best]+127;
  for (int j=0;j<width;j+=2)
    output[j/2]=uint8_t((values[key[j]]&15) | ((values[key[j+1]]&15)<<4));
}
}
