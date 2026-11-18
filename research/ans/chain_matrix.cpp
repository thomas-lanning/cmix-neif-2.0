// SPDX-License-Identifier: GPL-3.0-only
// Reuse the release predictor and carry the rANS state across matrix boundaries.
#include "residual_matrix.h"
#include <cstdio>
#include <cstring>

using namespace matrix_ans;
int main(int argc,char** argv) {
  if (argc!=8) return 2;
  size_t rows=std::strtoul(argv[2],nullptr,10), cols=std::strtoul(argv[3],nullptr,10);
  unsigned mode=std::strtoul(argv[6],nullptr,10);
  uint32_t state=uint32_t(std::strtoul(argv[7],nullptr,10));
  check(rows && cols && rows<=65536 && cols<=65536 && rows*cols<=16000000 && mode<48 && state>=lower);
  FILE* in=std::fopen(argv[4],"rb");
  FILE* out=std::fopen(argv[5],"wb");
  check(in && out);
  std::vector<int8_t> q(rows*cols);
  if (!std::strcmp(argv[1],"encode")) {
    check(std::fread(q.data(),1,q.size(),in)==q.size() && std::getc(in)==EOF);
    for (int x:q) check(std::abs(x)<=7);
    Writer writer;
    selected_pass(q.data(),rows,cols,mode,writer);
    auto encoded=writer.finish(state);
    check(std::fwrite(encoded.data(),1,encoded.size(),out)==encoded.size());
  } else if (!std::strcmp(argv[1],"decode")) {
    check(std::fseek(in,0,SEEK_END)==0);
    const long size=std::ftell(in);check(size>=0 && std::fseek(in,0,SEEK_SET)==0);
    std::vector<uint8_t> data(4+size);
    for (unsigned j=0;j<4;++j) data[j]=uint8_t(state>>(8*j));
    check(std::fread(data.data()+4,1,size,in)==size_t(size));
    Reader reader(data.data(),data.size());
    selected_pass(q.data(),rows,cols,mode,reader);
    uint32_t consumed=uint32_t(reader.p-data.data()-4);
    check(std::fwrite(&reader.state,4,1,out)==1 && std::fwrite(&consumed,4,1,out)==1);
    check(std::fwrite(q.data(),1,q.size(),out)==q.size());
  } else return 2;
  check(!std::ferror(in) && std::fclose(in)==0 && std::fclose(out)==0);
}
