// SPDX-License-Identifier: GPL-3.0-only
#include "residual_matrix.h"
#include <cstdio>
#include <cstring>

using namespace matrix_ans;
int main(int argc,char** argv) {
  if (argc!=6) return 2;
  size_t rows=std::strtoul(argv[2],nullptr,10),cols=std::strtoul(argv[3],nullptr,10);
  check(rows && cols && rows<=65536 && cols<=65536 && rows*cols<=16000000);
  FILE* in=std::fopen(argv[4],"rb");
  check(in!=nullptr);
  std::vector<int8_t> q(rows*cols);
  std::vector<uint8_t> encoded;
  if (!std::strcmp(argv[1],"encode")) {
    check(std::fread(q.data(),1,q.size(),in)==q.size() && std::getc(in)==EOF);
    for(int v:q) check(v>=-7 && v<=7);
    std::array<std::vector<RowChoice>,2> choices;
    for (unsigned mode=0; mode<48; ++mode) {
      if (mode>=32 && (rows>1024 || cols>1024 || rows<16 || cols<16)) continue;
      if (mode<16 && ((mode&7)==4 || (mode&7)==6)) continue;
      Writer writer;
      writer.row_choices=&choices;
      selected_pass(q.data(),rows,cols,mode,writer);
      auto trial=writer.finish();
      if (encoded.empty() || trial.size()+1<encoded.size()) {
        encoded.swap(trial);encoded.insert(encoded.begin(),uint8_t(mode));
      }
    }
    auto recovered=std::vector<int8_t>(q.size());
    Reader reader(encoded.data()+1,encoded.size()-1);
    selected_pass(recovered.data(),rows,cols,encoded[0],reader);reader.finish();
    check(recovered==q);
  } else if (!std::strcmp(argv[1],"decode")) {
    int ch;
    while ((ch=std::getc(in))!=EOF) encoded.push_back(uint8_t(ch));
    check(encoded.size()>=5 && encoded[0]<48);
    Reader reader(encoded.data()+1,encoded.size()-1);
    selected_pass(q.data(),rows,cols,encoded[0],reader);reader.finish();
  } else return 2;
  check(!std::ferror(in) && std::fclose(in)==0);
  FILE* out=std::fopen(argv[5],"wb");
  check(out!=nullptr);
  if (!std::strcmp(argv[1],"encode")) check(std::fwrite(encoded.data(),1,encoded.size(),out)==encoded.size());
  else check(std::fwrite(q.data(),1,q.size(),out)==q.size());
  check(std::fclose(out)==0);
}
