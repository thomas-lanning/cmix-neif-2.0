"""Install the frozen five-coefficient correction in both release coder paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from install import replace

HERE=Path(__file__).resolve().parent


def install(root):
    record=json.loads((HERE / "frozen-mixer.json").read_bytes())
    model=record["model"]
    if model["scope"]!="global" or not model["stages"] or model["scale"]!=1024 or model["coefficient_bytes"]!=10:
        raise ValueError("unsupported fitted correction")
    coeff=model["coefficients"][0]
    if len(coeff)!=5 or any(type(c)!=int or c < -32768 or c > 32767 for c in coeff):
        raise ValueError("invalid coefficients")
    cpp="""#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
extern bool research_body_mixer;
static inline double mixer_logit(uint32_t p) {
  return std::log(double(p)/double((uint64_t{1}<<32)-p));
}
static inline uint32_t mixer_p(uint32_t final_p,uint32_t floor_p,uint32_t head_p,uint32_t base_p) {
  if (!research_body_mixer) return final_p;
  const double l=mixer_logit(final_p);
  const double f=floor_p==final_p ? l : mixer_logit(floor_p);
  const double h=head_p==floor_p ? f : mixer_logit(head_p);
  const double b=base_p==head_p ? h : mixer_logit(base_p);
  static constexpr int16_t w[]={COEFFICIENTS};
  const double change=w[0]+w[1]*std::clamp(l,-15.,15.)+w[2]*std::clamp(f-l,-6.,6.)
      +w[3]*std::clamp(h-f,-6.,6.)+w[4]*std::clamp(b-h,-6.,6.);
  return uint32_t(std::clamp(std::nearbyint(4294967296./(1.+std::exp(-l-change/1024.))),1.,4294967295.));
}
"""
    coder=root/"src/coder"
    (coder/"research_mixer.h").write_text(cpp.replace("COEFFICIENTS",",".join(map(str,coeff))))
    for name in ("encoder","decoder"):
        f=coder/(name+".cpp")
        s=replace(f.read_text(),"  int bb_bitl = 0;","  const uint32_t base_p=uint32_t(p)<<16;\n  int bb_bitl = 0;")
        s=replace(s,"  const unsigned int xmid = floor_stage::Mid(x1_, x2_, p_->SemCorrect(floor_.P32(p,",
                    "  const uint32_t floor_p=floor_.P32(p,")
        s=replace(s,"(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7)));","""(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7);
  const uint32_t final_p=p_->SemCorrect(floor_p);
  const uint32_t adjusted=mixer_p(final_p,floor_p,uint32_t(p)<<16,base_p);
  const unsigned int xmid=floor_stage::Mid(x1_,x2_,adjusted);""")
        f.write_text('#include "research_mixer.h"\n'+s)
    f=root/"src/runner.cpp"
    s=f.read_text()
    # All bytes before the first article body retain the authenticated stream.
    for indent in (6,4):
        line=" "*indent+"p->SemByteStart(sem_pb);"
        s=replace(s,"\n"+line,"\n"+" "*indent+"research_body_mixer=(pos>=15282108);\n"+line)
    f.write_text("bool research_body_mixer=false;\n"+s)


def bounded(root):
    subprocess.run([sys.executable,HERE.parent/"quant/instrument_prefix.py",root,"--no-events"],check=True)
    install(root)
    f=root/"src/runner.cpp"
    s=replace(f.read_text(),"  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);",
              "  census_block_active=true;\n  isbn_check::after_at.an=true;\n  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);")
    s=replace(s,"    putc(byte, os);\n    sem_pb","""    putc(byte, os);
    if (pos==19999999) std::exit(fflush(os) ? 4 : 0);
    sem_pb""")
    f.write_text(s)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("--bounded",action="store_true");a=p.parse_args()
    (bounded if a.bounded else install)(a.root)
