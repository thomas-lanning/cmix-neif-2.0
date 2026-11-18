// SPDX-License-Identifier: GPL-3.0-only
// Research-only capture. The scored reader needs none of these tables or files.
#pragma once
namespace cache_trace {
using namespace semantic_cache;
static int8_t* shadow;
static FILE* pending;
static unsigned population[3],selected[3];
static FILE* output[3];
static void put(FILE* file,const void* data,size_t n) {
  if (!file || std::fwrite(data,1,n,file)!=n) std::abort();
}
__attribute__((noinline)) static void store(uint32_t ri,const int8_t* key) {
  if (!shadow) {
    shadow=static_cast<int8_t*>(std::calloc(size_t(kN),kD));
    if (!shadow) std::abort();
  }
  std::memcpy(shadow+size_t(ri)*kD,key,kD);
}
__attribute__((noinline)) static void target(uint8_t symbol,uint64_t pos) {
  if (pending) {put(pending,&symbol,1);pending=nullptr;}
  if (pos==18000000 || pos==30000000 || pos==40000000) {
    unsigned i=pos==18000000?0:pos==30000000?1:2;
    if (output[i]) {if(std::fclose(output[i]))std::abort();output[i]=nullptr;}
    char name[64];
    std::snprintf(name,sizeof name,"cache-%u.done",unsigned(pos-2000000));
    FILE* f=std::fopen(name,"w");
    if (!f || std::fprintf(f,"%u %u\n",population[i],selected[i])<0 || std::fclose(f)) std::abort();
  }
}
__attribute__((noinline)) static void sample(uint64_t pos,const State& s,
        const uint32_t* ids,int n,const uint32_t* top,const double* similarities,int k,const float* transformer) {
  constexpr uint32_t starts[]={16000000,28000000,38000000};
  int group=-1;
  for (int i=0;i<3;++i) if (pos>=starts[i] && pos<starts[i]+2000000u) group=i;
  if (group<0 || n<8) return;
  ++population[group];
  if (selected[group]>=1536 || (population[group]%97)!=1) return;
  ++selected[group];
  FILE*& f=output[group];
  if (!f) {
    char name[64];std::snprintf(name,sizeof name,"cache-%u.bin",starts[group]);
    f=std::fopen(name,"wb");put(f,"NCACHE01",8);
  }
  uint32_t meta[]={uint32_t(pos),uint32_t(n),uint32_t(k)};
  put(f,meta,sizeof meta);put(f,s.ck,kD);put(f,transformer,256*sizeof(float));
  for (int j=0;j<n;++j) {
    uint32_t ri=ids[j]&(kN-1);
    put(f,&ri,4);put(f,&s.val[ri],1);put(f,shadow+size_t(ri)*kD,kD);
    put(f,s.key+size_t(ri)*kKB,kKB);
  }
  put(f,top,4*k);put(f,similarities,sizeof(double)*k);
  pending=f;
}
}
