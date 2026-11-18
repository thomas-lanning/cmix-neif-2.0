// SPDX-License-Identifier: GPL-3.0-only
// One byte-renormalized rANS state over a forward trace, using bounded RAM.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

constexpr uint64_t low = uint64_t{1} << 40;
constexpr size_t block = 65536;

void check(bool good, const char* error) {
  if (!good) throw std::runtime_error(error);
}

uint32_t read_p(const uint8_t* data) {
  return uint32_t(data[0]) | (uint32_t(data[1]) << 8) |
         (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
}

uint64_t frequency(uint32_t p, unsigned bits) {
  const uint64_t total = uint64_t{1} << bits;
  return std::clamp((uint64_t(p)*total + (uint64_t{1} << 31)) >> 32, uint64_t{1}, total-1);
}

uint64_t run(FILE* trace, FILE* suffix, FILE* output, unsigned bits, uint64_t count) {
  const uint64_t total = uint64_t{1} << bits;
  std::array<uint8_t,5*block> data;
  uint64_t state=low;
  for (uint64_t end=count; end;) {
    size_t n=std::min<uint64_t>(end,block);
    end-=n;
    check(fseeko(trace,off_t(end*5),SEEK_SET)==0 && fread(data.data(),5,n,trace)==n,"trace read");
    for (size_t i=n; i--;) {
      const uint8_t* row=&data[i*5];
      uint32_t p=read_p(row);
      check(p && row[4]<=1,"invalid decision");
      uint64_t f=frequency(p,bits);
      uint64_t start=row[4] ? 0 : f;
      uint64_t width=row[4] ? f : total-f;
      while (state>=((low>>bits)<<8)*width) {
        check(fputc(int(state&255),suffix)!=EOF,"reverse spool write");
        state>>=8;
      }
      state=((state/width)<<bits)+state%width+start;
    }
  }
  check(state>=low && state<(low<<8),"invalid encoded state");
  // All normalized states fit in six bytes. The containing stream supplies
  // the stopping condition, so there is no length field between RAM chunks.
  for (unsigned i=0;i<6;++i) check(fputc(int((state>>(8*i))&255),output)!=EOF,"state write");
  off_t length=ftello(suffix);
  check(length>=0,"reverse spool length");
  for (off_t end=length;end;) {
    size_t n=std::min<uint64_t>(end,data.size());
    end-=n;
    check(fseeko(suffix,end,SEEK_SET)==0 && fread(data.data(),1,n,suffix)==n,"reverse spool read");
    std::reverse(data.begin(),data.begin()+n);
    check(fwrite(data.data(),1,n,output)==n,"output write");
  }
  check(fflush(output)==0 && fseeko(output,0,SEEK_SET)==0 && fseeko(trace,0,SEEK_SET)==0,"rewind");
  state=0;
  for (unsigned i=0;i<6;++i) {
    int ch=fgetc(output); check(ch!=EOF,"missing rANS state"); state|=uint64_t(ch)<<(8*i);
  }
  for (uint64_t pos=0;pos<count;) {
    size_t n=std::min<uint64_t>(count-pos,block);
    check(fread(data.data(),5,n,trace)==n,"forward trace read");
    for (size_t i=0;i<n;++i) {
      const uint8_t* row=&data[i*5];
      uint64_t f=frequency(read_p(row),bits);
      uint64_t slot=state&(total-1);
      bool bit=slot<f;
      check(bit==bool(row[4]),"decoded bit differs");
      state=(bit?f:total-f)*(state>>bits)+slot-(bit?0:f);
      while (state<low) {
        int ch=fgetc(output); check(ch!=EOF,"truncated rANS payload"); state=(state<<8)|uint64_t(ch);
      }
    }
    pos+=n;
  }
  check(state==low && fgetc(output)==EOF && !ferror(output),"invalid rANS termination");
  return uint64_t(length)+6;
}

int main(int argc,char** argv) {
  if (argc!=4) return 2;
  try {
    unsigned bits=unsigned(std::stoul(argv[3]));
    check(bits>=1 && bits<=32,"use probability precision 1 through 32");
    FILE* trace=fopen(argv[1],"rb");
    check(trace && fseeko(trace,0,SEEK_END)==0,"open trace");
    off_t bytes=ftello(trace);
    check(bytes>0 && bytes%5==0,"trace must contain complete decisions");
    FILE* suffix=tmpfile();
    check(suffix!=nullptr,"create private reverse spool");
    FILE* output=fopen(argv[2],"w+bx");
    check(output!=nullptr,"choose a fresh output file");
    uint64_t size=run(trace,suffix,output,bits,uint64_t(bytes)/5);
    check(fclose(output)==0 && fclose(suffix)==0 && fclose(trace)==0,"close files");
    std::printf("{\"events\":%llu,\"precision\":%u,\"bytes\":%llu,\"state_bytes\":6,"
                "\"bit_roundtrip\":true,\"trace_bytes\":%llu,\"reverse_spool_bytes\":%llu}\n",
                (unsigned long long)(bytes/5),bits,(unsigned long long)size,
                (unsigned long long)bytes,(unsigned long long)(size-6));
  } catch (const std::exception& e) {
    std::fprintf(stderr,"%s\n",e.what()); return 1;
  }
}
