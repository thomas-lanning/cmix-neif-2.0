// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <array>
#include <cstdint>

namespace cache_int3 {
constexpr int8_t levels[]={-6,-3,-1,0,1,2,3,6};
struct Lookup {
  uint8_t index[5][255]{};
  Lookup() {
    const int gain[]={72,90,100,115,140};
    for (int g=0;g<5;++g) for (int raw=-127;raw<=127;++raw) {
      int best=0,dist=1<<30;
      for(int j=0;j<8;++j) {
        int err=raw*gain[g]*6-int(levels[j])*12700;
        if(err<0)err=-err;
        if(err<dist) {dist=err;best=j;}
      }
      index[g][raw+127]=uint8_t(best);
    }
  }
};
inline void pack(uint8_t* out,const int8_t* key) {
  static const Lookup lut;
  int best=0;
  int64_t best_dot=-1,best_norm=1;
  for (int g=0;g<5;++g) {
    int64_t dot=0,norm=0;
    for (int j=0;j<128;++j) {
      int v=levels[lut.index[g][int(key[j])+127]];
      dot+=v*int(key[j]);norm+=v*v;
    }
    if(norm && dot*dot*best_norm>best_dot*norm) {
      best=g;best_dot=dot*dot;best_norm=norm;
    }
  }
  for (int b=0;b<16;++b) {
    uint32_t word=0;
    for (int i=0;i<8;++i) word|=uint32_t(lut.index[best][int(key[8*b+i])+127])<<(3*i);
    for (int i=0;i<3;++i) out[3*b+i]=uint8_t(word>>(8*i));
  }
}
inline void dot(const uint8_t* packed,const int8_t* query,int32_t& product,uint32_t& norm) {
  product=0;norm=0;
  for (int b=0;b<16;++b) {
    const uint8_t* p=packed+3*b;
    uint32_t word=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16);
    for (int i=0;i<8;++i) {
      int v=levels[(word>>(3*i))&7];
      product+=v*int(query[8*b+i]);norm+=unsigned(v*v);
    }
  }
}
}
