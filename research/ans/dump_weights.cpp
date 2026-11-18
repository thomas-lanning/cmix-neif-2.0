// SPDX-License-Identifier: GPL-3.0-only
// Compile against the corresponding loader. Compare stdout for two containers.
#include "weights_io.h"
#include <cstdio>
int main(int argc,char** argv) {
  if (argc!=2) return 2;
  auto w=fx2::WeightsFile::load_compressed(argv[1]);
  unsigned found=0;
  for (int i=0;i<fx2::namen::kN*17;++i) {
    const auto& t=w.t[i];
    if (!t.data) continue;
    ++found;
    const size_t width=t.dtype==fx2::DT_I8?1:t.dtype==fx2::DT_BF16?2:4;
    if (std::fwrite(&i,sizeof i,1,stdout)!=1 ||
        std::fwrite(&t.dtype,1,1,stdout)!=1 || std::fwrite(&t.ndim,1,1,stdout)!=1 ||
        std::fwrite(t.shape,sizeof t.shape[0],t.ndim,stdout)!=t.ndim ||
        std::fwrite(t.data,width,t.numel,stdout)!=t.numel) return 3;
  }
  return found==597 && std::fflush(stdout)==0?0:4;
}
