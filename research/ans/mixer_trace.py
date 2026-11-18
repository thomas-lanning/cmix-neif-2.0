"""Capture the final mixer's successive probabilities on disjoint canonical spans."""
from pathlib import Path
import subprocess
import sys

from install import replace

SPANS=(16_000_000,20_000_000,26_000_000,34_000_000,42_000_000)
LENGTH=262_144


def instrument(root):
    here=Path(__file__).resolve().parent
    subprocess.run([sys.executable,here.parent/"quant/instrument_prefix.py",root,"--no-events"],check=True)
    header=root/"src/coder/encoder.h"
    text=replace(header.read_text(),"  void Flush();","""  void Flush();
  void PublishPrefix() {
    if (fwrite(out_,1,out_n_,os_) != out_n_ || fflush(os_)) std::exit(6);
    total_out_ += out_n_; out_n_ = 0;
  }""")
    header.write_text("#include <cstdlib>\n"+text)
    runner=root/"src/runner.cpp"
    text=runner.read_text()
    text=replace(text,"200000 : 20000000;",f"200000 : {SPANS[-1]+LENGTH};")
    hook="""
      for (unsigned offset : research_offsets) {
        if (pos == offset) {
          char path[80];
          snprintf(path,sizeof(path),"span-%u.bin",offset);
          research_trace = fopen(path,"wbx");
          if (!research_trace) std::exit(6);
        }
      }
      research_previous = (unsigned char)sem_pb;
"""
    text=replace(text,"      p->SemByteStart(sem_pb);",hook+"      p->SemByteStart(sem_pb);")
    hook="""
      if (research_trace) {
        for (unsigned offset : research_offsets) {
          if (pos == offset + RESEARCH_SPAN_LENGTH) {
            if (fclose(research_trace)) std::exit(6);
            research_trace = nullptr;
            e.PublishPrefix();
            FILE* report = fopen("spans.tsv", "a");
            if (!report) std::exit(6);
            fprintf(report,"%u\\t%llu\\t%zu\\n",offset,pos,e.OutputSize()+5);
            if (fclose(report)) std::exit(6);
          }
        }
      }
"""
    text=replace(text,"      if (pos == stop) {",hook+"      if (pos == stop) {")
    runner.write_text("#include <cstdio>\nextern FILE* research_trace;\nextern unsigned char research_previous;\n"
        +"static constexpr unsigned research_offsets[] = {"+",".join(map(str,SPANS))+"};\n"
        +f"static constexpr unsigned RESEARCH_SPAN_LENGTH = {LENGTH};\n"+text)
    file=root/"src/coder/encoder.cpp"
    text=replace(file.read_text(),"  int bb_bitl = 0;","  const uint32_t base_p = uint32_t(p)<<16;\n  int bb_bitl = 0;")
    text=replace(text,"  const unsigned int xmid = floor_stage::Mid(x1_, x2_, p_->SemCorrect(floor_.P32(p,",
                      "  const unsigned int floor_p = floor_.P32(p,")
    tail="(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7)));"
    text=replace(text,tail,"""(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7);
  const uint32_t final_p = p_->SemCorrect(floor_p);
  if (research_trace) research_record(final_p, floor_p, uint32_t(p)<<16, base_p, p_->BbBitCtx(), bit);
  const unsigned int xmid = floor_stage::Mid(x1_, x2_, final_p);""")
    anchor="  if (p > 65534) p = 65534;"
    text=replace(text,anchor,anchor+"""
  if (research_trace) {
    const uint32_t rp = uint32_t(p)<<16;
    research_record(rp,rp,rp,rp,0,bit);
  }""")
    file.write_text("""#include <cstdint>
#include <cstdio>
#include <cstdlib>
FILE* research_trace = nullptr;
unsigned char research_previous = 0;
struct ResearchEvent { uint32_t final_p, floor_p, head_p, base_p; uint16_t prefix; uint8_t prev, bit; };
static_assert(sizeof(ResearchEvent)==20, "event ABI");
static void research_record(uint32_t f,uint32_t s,uint32_t h,uint32_t b,unsigned ctx,int bit) {
  if (!f || !s || !h || !b || ctx>65535 || bit<0 || bit>1) std::exit(6);
  const ResearchEvent e{f,s,h,b,uint16_t(ctx),research_previous,uint8_t(bit)};
  if (fwrite(&e,sizeof(e),1,research_trace)!=1) std::exit(6);
}
"""+text)


if __name__=="__main__":
    instrument(Path(sys.argv[1]).resolve())
