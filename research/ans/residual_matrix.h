// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <type_traits>
#include "adaptive_matrix.h"
namespace residual_ans {
using namespace matrix_ans;
inline unsigned bounded(unsigned p) {return std::clamp(p,1u,total-1);}
struct Residual {
  unsigned n=0,z=0;
  uint32_t expected=0;
  void update(unsigned bit,unsigned base) {
    ++n;z+=!bit;expected+=base;
    if (n>=4096) {n/=2;z=(z+1)/2;expected/=2;}
  }
  unsigned predict(unsigned p,unsigned strength) const {
    unsigned actual=bounded((z*total+strength*p)/(n+strength));
    unsigned avg=bounded((expected+strength*p)/(n+strength));
    return bounded(odds(p,actual,avg));
  }
};
using RTree=std::array<Residual,19>;
using CTree=std::array<Count,19>;
// Condition the next bit on the discrepancy between predicted and observed row/column counts.
template<class Codec>
void pass(int8_t* weights,size_t rows,size_t cols,unsigned mode,Codec& codec) {
  const bool transpose=mode&8;
  if(transpose)std::swap(rows,cols);
  const unsigned variant=mode&7;
  const unsigned prior_strength=(variant&2)?96:24;
  const std::array<unsigned,8> order=(variant&1)?std::array<unsigned,8>{0,7,1,2,3,4,5,6}:std::array<unsigned,8>{0,1,2,3,4,5,6,7};
  unsigned index[8];for(unsigned i=0;i<8;++i)index[order[i]]=i;
  CTree global{};std::array<CTree,24> width_ctx{};
  Count empty_rows{};
  std::vector<RTree> col(cols);std::vector<unsigned> col_abs(cols);
  std::vector<CTree> lag(16);
  uint64_t abs_sum=0,count=0;
  for(size_t i=0;i<rows;++i) {
    const auto offset=[&](size_t j){return transpose?j*rows+i:i*cols+j;};
    bool all_zero=true;
    if constexpr (std::is_same_v<Codec, Writer>) {
      for(size_t j=0;j<cols;++j) if(weights[offset(j)]){all_zero=false;break;}
    }
    unsigned empty=codec.bit(bounded(empty_rows.predict(total-32,2)),all_zero);
    empty_rows.update(empty);
    RTree row{}; unsigned row_abs=0;int previous=0;
    for(size_t j=0;j<cols;++j) {
      size_t at=offset(j);
      if(empty){weights[at]=0;continue;}
      int raw=weights[at];unsigned mag=unsigned(std::abs(raw));
      unsigned mean=unsigned((abs_sum*256+64*742)/(count+64));
      unsigned r=unsigned((row_abs*256+16*mean)/(j+16));
      unsigned c=unsigned((col_abs[j]*256+16*mean)/(i+16));
      auto width=std::clamp(unsigned(8ull*r*c/(std::max(1u,mean)*256)),8u,31u)-8;
      auto event=[&](unsigned node,unsigned observed){
        auto& g=global[node];auto& w=width_ctx[width][node];
        auto& l=lag[previous+7][node];
        unsigned base=bounded(w.predict(g.predict(total/2,4),48));
        if(variant&4)base=bounded((3*base+l.predict(base,64))/4);
        unsigned pr=row[node].predict(base,prior_strength);
        unsigned pc=col[j][node].predict(base,prior_strength);
        unsigned p=bounded((3*odds(pr,pc,base)+base)/4);
        unsigned b=codec.bit(p,observed);
        g.update(b);w.update(b);l.update(b);row[node].update(b,base);col[j][node].update(b,base);
        return b;
      };
      // Predict zero separately. All remaining symbols use a truncated tree.
      unsigned zero=event(1,mag==0); unsigned symbol=0;
      if(!zero){
        unsigned v=index[mag]-1;
        unsigned high=event(2,v>=3);
        if(high){unsigned a=event(3,(v-3)>>1&1);unsigned b=event(4+a,(v-3)&1);symbol=4+2*a+b;}
        else {unsigned a=event(6,v!=0);symbol=1+a;if(a)symbol+=event(7,v==2);}
      }
      mag=order[symbol];
      unsigned neg=mag?event(8+mag,raw<0):0;
      weights[at]=previous=neg?-int(mag):int(mag);
      row_abs+=mag;col_abs[j]+=mag;abs_sum+=mag;++count;
    }
  }
}
}

namespace matrix_ans {
template<class Codec>
void selected_pass(int8_t* q, size_t rows, size_t cols, unsigned mode, Codec& codec) {
  if (mode < 16) matrix_ans::pass(q, rows, cols, mode, codec);
  else residual_ans::pass(q, rows, cols, mode-16, codec);
}
}
