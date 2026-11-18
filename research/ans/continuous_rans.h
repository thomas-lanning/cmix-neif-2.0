// SPDX-License-Identifier: GPL-3.0-only
// One state for the entire adaptive stream. The forward model decisions live in
// a private temporary file. Both reverse passes use a fixed memory budget.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>

namespace model_rans {
constexpr uint64_t lower=uint64_t{1}<<55, total=uint64_t{1}<<32;
constexpr size_t rows_per_chunk=65536;
inline void require(bool ok) { if (!ok) {std::fputs("continuous rANS I/O or state error\n",stderr);std::exit(6);} }
class Encoder {
  FILE* trace=std::tmpfile();
  uint64_t count=0;
  size_t written=0;
 public:
  Encoder() {require(trace);}
  Encoder(const Encoder&)=delete;
  ~Encoder(){if (trace) std::fclose(trace);}
  void push(FILE*,uint32_t p,int bit) {
    require(trace && p && (bit==0 || bit==1));
    uint8_t row[5]={uint8_t(p),uint8_t(p>>8),uint8_t(p>>16),uint8_t(p>>24),uint8_t(bit)};
    require(std::fwrite(row,sizeof row,1,trace)==1);
    ++count;
  }
  size_t size() const {return written;}
  void flush(FILE* file) {
    require(trace && std::fflush(trace)==0);
    FILE* suffix=std::tmpfile();require(suffix);
    std::array<uint8_t,5*rows_per_chunk> buf;
    uint64_t state=lower;
    for (uint64_t end=count;end;) {
      size_t n=std::min<uint64_t>(end,rows_per_chunk);end-=n;
      require(fseeko(trace,off_t(end*5),SEEK_SET)==0 && std::fread(buf.data(),5,n,trace)==n);
      for (size_t i=n;i--;) {
        const uint8_t* row=&buf[5*i];
        uint64_t p=uint32_t(row[0])|(uint32_t(row[1])<<8)|(uint32_t(row[2])<<16)|(uint32_t(row[3])<<24);
        uint64_t width=row[4]?p:total-p, start=row[4]?0:p;
        while (state>=((lower>>32)<<8)*width) {
          require(std::putc(int(state&255),suffix)!=EOF);state>>=8;
        }
        state=((state/width)<<32)+state%width+start;
      }
    }
    require(std::fclose(trace)==0);trace=nullptr;
    for (unsigned j=0;j<8;++j) require(std::putc(int((state>>(8*j))&255),file)!=EOF);
    off_t bytes=ftello(suffix);require(bytes>=0);
    for (off_t end=bytes;end;) {
      size_t n=std::min<uint64_t>(end,buf.size());end-=n;
      require(fseeko(suffix,end,SEEK_SET)==0 && std::fread(buf.data(),1,n,suffix)==n);
      std::reverse(buf.begin(),buf.begin()+n);
      require(std::fwrite(buf.data(),1,n,file)==n);
    }
    written=size_t(bytes)+8;require(std::fclose(suffix)==0);
  }
};
class Decoder {
  uint64_t state=0;
  static unsigned next_byte(FILE* file) {int c=std::getc(file);require(c!=EOF);return unsigned(c);}
 public:
  int next(FILE* file,uint32_t p) {
    require(p!=0);
    if (!state) {
      for (unsigned j=0;j<8;++j) state|=uint64_t(next_byte(file))<<(8*j);
      require(state>=lower && state<(lower<<8));
    }
    uint64_t slot=uint32_t(state);
    int bit=(slot<p);
    state=(bit?p:total-p)*(state>>32)+slot-(bit?0:p);
    while (state<lower) state=(state<<8)|next_byte(file);
    return bit;
  }
  void finish(FILE* file) {
    require(state==lower && std::getc(file)==EOF && !std::ferror(file));
  }
};
}
