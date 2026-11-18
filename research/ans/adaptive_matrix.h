// SPDX-License-Identifier: GPL-3.0-only
// Binary rANS with causal row, column and magnitude contexts for signed weights.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace matrix_ans {
constexpr unsigned total=32768, lower=1u<<23;
inline void check(bool ok) { if (!ok) std::abort(); }
struct Count {
  unsigned zero=0,n=0;
  unsigned predict(unsigned prior,unsigned strength) const {
    return (zero*total+prior*strength)/(n+strength);
  }
  void update(unsigned bit) {
    zero+=!bit;
    if (++n==4096) {zero=(zero+1)/2;n/=2;}
  }
};
using Tree=std::array<Count,16>;

struct Writer {
  std::vector<uint16_t> events;
  unsigned bit(unsigned p,unsigned actual) {
    events.push_back(uint16_t(2*p+actual));
    return actual;
  }
  std::vector<uint8_t> finish() const {
    uint32_t state=lower;
    std::vector<uint8_t> out;
    for (auto it=events.rbegin();it!=events.rend();++it) {
      unsigned p=*it>>1, one=*it&1;
      unsigned start=one?p:0, freq=one?total-p:p;
      while (state >= ((lower/total)<<8)*freq) {out.push_back(uint8_t(state));state>>=8;}
      state=(state/freq)*total+state%freq+start;
    }
    for (int i=3;i>=0;--i) out.push_back(uint8_t(state>>(8*i)));
    std::reverse(out.begin(),out.end());
    return out;
  }
};
struct Reader {
  const uint8_t *p,*end;
  uint32_t state=0;
  Reader(const uint8_t* data,size_t n):p(data),end(data+n) {
    check(n>=4);
    for (unsigned k=0;k<4;++k) state|=uint32_t(*p++)<<(8*k);
    check(state>=lower);
  }
  unsigned bit(unsigned prob,unsigned) {
    unsigned slot=state%total,one=slot>=prob;
    unsigned freq=one?total-prob:prob;
    state=freq*(state/total)+slot-(one?prob:0);
    while (state<lower) {check(p<end);state=(state<<8)|*p++;}
    return one;
  }
  void finish() const {check(state==lower && p==end);}
};

inline unsigned odds(unsigned first,unsigned second,unsigned prior) {
  uint64_t a=uint64_t(first)*second*(total-prior);
  uint64_t b=uint64_t(total-first)*(total-second)*prior;
  return unsigned((a*total)/(a+b));
}

template<class Coder>
void pass(int8_t* q,size_t rows,size_t cols,unsigned mode,Coder& codec) {
  const bool transpose=mode&8;
  const unsigned variant=mode&7;
  if (transpose) std::swap(rows,cols);
  std::vector<Tree> by_col(cols);
  std::vector<unsigned> col_sum(cols);
  Tree global{};
  std::array<Tree,24> by_width{};
  std::array<Tree,96> by_peak{};
  uint64_t abs_sum=0,n=0;
  for (size_t i=0;i<rows;++i) {
    Tree by_row{};
    unsigned row_sum=0,peak=0;
    for (size_t j=0;j<cols;++j) {
      size_t offset=transpose?j*rows+i:i*cols+j;
      int original=q[offset];
      unsigned mag=unsigned(std::abs(original));
      unsigned mean=unsigned((abs_sum*256+64*742)/(n+64));
      unsigned r=unsigned((row_sum*256+24*mean)/(j+24));
      unsigned c=unsigned((col_sum[j]*256+24*mean)/(i+24));
      unsigned scale=unsigned((8ull*r*c)/(std::max(1u,mean)*256));
      unsigned w=std::clamp(scale,8u,31u)-8;
      unsigned tier=peak>3?3:peak>1?2:peak>0?1:0;
      auto code=[&](unsigned node,unsigned observed) {
        auto& g=global[node]; auto& b=by_width[w][node];
        auto& t=by_peak[4*w+tier][node];
        auto& row=by_row[node]; auto& col=by_col[j][node];
        unsigned base=g.predict(total/2,2);
        unsigned width=b.predict(base,48);
        unsigned prior=(variant&2)?t.predict(width,96):width;
        unsigned prediction=prior;
        if (variant&1) {
          unsigned strength=(variant&4)?96:32;
          unsigned pr=row.predict(prior,strength);
          unsigned pc=col.predict(prior,strength);
          prediction=(3*odds(pr,pc,std::clamp(prior,1u,total-1))+prior)/4;
        }
        unsigned value=codec.bit(std::clamp(prediction,1u,total-1),observed);
        g.update(value);b.update(value);t.update(value);row.update(value);col.update(value);
        return value;
      };
      unsigned node=1;
      for (int k=2;k>=0;--k) node=2*node+code(node,(mag>>k)&1);
      mag=node-8;
      unsigned neg=mag?code(7+mag,original<0):0;
      q[offset]=int8_t(neg?-int(mag):int(mag));
      row_sum+=mag;col_sum[j]+=mag;abs_sum+=mag;++n;
      peak=std::max(peak,mag);
    }
  }
}
}
