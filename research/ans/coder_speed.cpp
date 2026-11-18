// SPDX-License-Identifier: GPL-3.0-only
// Time only the entropy coder on the same recorded model probabilities.
#include "model_rans.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock=std::chrono::steady_clock;
struct Event { uint32_t p; uint8_t bit; };
struct ArithmeticEncoder {
  uint32_t lo=0,hi=UINT32_MAX;
  void push(FILE* file,uint32_t p,int bit) {
    uint32_t middle=lo+((uint64_t(hi-lo)*p)>>32);
    if (bit) hi=middle; else lo=middle+1;
    while ((lo>>24)==(hi>>24)) {
      model_rans::require(std::putc(hi>>24,file)!=EOF);
      lo<<=8;hi=(hi<<8)|255;
    }
  }
  void flush(FILE* file) { model_rans::require(std::putc(hi>>24,file)!=EOF); }
};
struct ArithmeticDecoder {
  uint32_t lo=0,hi=UINT32_MAX,code=0;
  bool initial=true;
  static unsigned read(FILE* stream) {
    int x=std::getc(stream);
    return x==EOF?0:unsigned(x);
  }
  int next(FILE* source,uint32_t p) {
    if (initial) {
      for (int i=0;i<4;++i) code=(code<<8)|read(source);
      initial=false;
    }
    uint32_t middle=lo+((uint64_t(hi-lo)*p)>>32);
    int bit=code<=middle;
    if (bit) hi=middle; else lo=middle+1;
    while ((lo>>24)==(hi>>24)) { lo<<=8;hi=(hi<<8)|255;code=(code<<8)|read(source); }
    return bit;
  }
  void finish(FILE* input) { model_rans::require(!std::ferror(input)); }
};
struct Result { long bytes;double encode,decode; };

template<class Encoder,class Decoder>
Result test(const std::vector<Event>& input) {
  FILE* f=std::tmpfile();
  model_rans::require(f!=nullptr);
  std::vector<char> buffer(1u<<20);
  model_rans::require(std::setvbuf(f,buffer.data(),_IOFBF,buffer.size())==0);
  Encoder e;Decoder d;
  auto start=Clock::now();
  for (auto row:input)e.push(f,row.p,row.bit);
  e.flush(f);
  model_rans::require(std::fflush(f)==0);
  auto encoded=Clock::now();
  long n=std::ftell(f);
  model_rans::require(n>=0 && std::fseek(f,0,SEEK_SET)==0);
  auto decode_start=Clock::now();
  for (auto row:input)model_rans::require(d.next(f,row.p)==row.bit);
  d.finish(f);
  auto end=Clock::now();
  model_rans::require(std::fclose(f)==0);
  return {n,std::chrono::duration<double>(encoded-start).count(),std::chrono::duration<double>(end-decode_start).count()};
}

double median(std::vector<double> v) {
  std::sort(v.begin(),v.end());
  return (v[v.size()/2]+v[(v.size()-1)/2])/2;
}

int main(int argc,char** argv) {
  if(argc!=2)return 2;
  FILE* f=std::fopen(argv[1],"rb");if (!f)return 2;
  std::vector<Event> rows;
  std::array<uint8_t,5> b;
  for (;;) {
    size_t n=std::fread(b.data(),1,b.size(),f);
    if(!n && std::feof(f))break;
    model_rans::require(n==5 && b[4]<=1);
    uint32_t p=b[0]|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);
    model_rans::require(p!=0);rows.push_back({p,b[4]});
  }
  model_rans::require(rows.size()>0 && std::fclose(f)==0);
  std::array<std::vector<double>,2> enc,dec;
  long sizes[2]={-1,-1};
  for(int rep=0;rep<10;++rep) for(int i=0;i<2;++i) {
    int coder=(rep+i)%2;
    auto r=coder?test<model_rans::Encoder,model_rans::Decoder>(rows):test<ArithmeticEncoder,ArithmeticDecoder>(rows);
    if(sizes[coder]>=0)model_rans::require(sizes[coder]==r.bytes);
    sizes[coder]=r.bytes;
    if(rep) {enc[coder].push_back(r.encode);dec[coder].push_back(r.decode);}
  }
  for(int i=0;i<2;++i) {
    double es=median(enc[i]),ds=median(dec[i]);
    for(auto& v:enc[i])v=std::abs(v-es);
    for(auto& v:dec[i])v=std::abs(v-ds);
    std::printf("{\"method\":\"%s\",\"events\":%zu,\"bytes\":%ld,\"repetitions\":9,"
                "\"encode_median_ms\":%.6f,\"decode_median_ms\":%.6f,"
                "\"encode_mad_ms\":%.6f,\"decode_mad_ms\":%.6f,\"roundtrip\":true}\n",
                i?"rans32":"arithmetic32",rows.size(),sizes[i],1000*es,1000*ds,1000*median(enc[i]),1000*median(dec[i]));
  }
}
