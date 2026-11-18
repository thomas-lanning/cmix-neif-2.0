"""Capture authenticated final-bit decisions and the release arithmetic state at block starts."""
from pathlib import Path
import sys

from install import replace

BLOCK=131072
COHORTS={"initial": ((118,120,122,128,132,136),18000000),
         "transfer": ((138,140,142,146,148,150),20000000)}
GROUPS,limit=COHORTS[sys.argv[2] if len(sys.argv)>2 else "initial"]
here=Path(sys.argv[1])
runner=here/"src/runner.cpp"
source=replace(runner.read_text(),"  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);",
               "  census_block_active=true;\n  isbn_check::after_at.an=true;\n  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);")
source=replace(source,"    int byte = 1;\n    p->SemByteStart(sem_pb);","""    int byte = 1;
    if (pos % 131072 == 0) calibration_block(pos / 131072);
    calibration_previous=uint8_t(sem_pb);
    p->SemByteStart(sem_pb);""")
source=replace(source,"    putc(byte, os);\n    sem_pb","""    putc(byte, os);
    if (pos == 17999999) {
      calibration_block(0);
      if (fflush(os)) std::exit(4);
      std::exit(0);
    }
    sem_pb""")
source=source.replace("if (pos == 17999999)",f"if (pos == {limit-1})")
runner.write_text("#include <cstdint>\nextern uint8_t calibration_previous;\nextern void calibration_block(unsigned);\n"+source)
file=here/"src/coder/decoder.cpp"
code=replace(file.read_text(),"  const unsigned int xmid = floor_stage::Mid(x1_, x2_, p_->SemCorrect(",
             "  const uint32_t initial_lo=x1_,initial_hi=x2_,head_p=uint32_t(p)<<16;\n  const uint32_t final_p=p_->SemCorrect(")
code=replace(code,"(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7)));",
             "(31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7));\n  const unsigned int xmid=floor_stage::Mid(x1_,x2_,final_p);")
code=replace(code,"  floor_.Update(bit);","  if (calibration_output) record(initial_lo,initial_hi,is_,final_p,head_p,p_->BbBitCtx(),bit);\n  floor_.Update(bit);")
code=replace(code,"  if (p > 65534) p = 65534;","  if (p > 65534) p = 65534;\n  const uint32_t initial_lo=x1_,initial_hi=x2_;")
a=code.index("int Decoder::DecodeRawBit")
part=replace(code[a:],"  while (((x1_^x2_)", "  if (calibration_output) record(initial_lo,initial_hi,is_,p<<16,p<<16,0,bit);\n  while (((x1_^x2_)")
code=code[:a]+part
header="""#include <cstdio>
#include <cstdlib>
#include <cstdint>
static FILE* calibration_output;
static bool first_event;
uint8_t calibration_previous;
void calibration_block(unsigned b) {
  if (calibration_output && std::fclose(calibration_output)) std::abort();
  calibration_output=nullptr;
  const unsigned blocks[]={BLOCK_IDS};
  for (unsigned i:blocks) if (b==i) {
    char name[64];std::snprintf(name,sizeof(name),"events-%u.bin",b);
    calibration_output=std::fopen(name,"wbx");if (!calibration_output) std::abort();
    first_event=true;
  }
}
static void record(uint32_t lo,uint32_t hi,FILE* input,uint32_t p,uint32_t head,unsigned prefix,int bit) {
  if (!p || prefix>65535 || !head) std::abort();
  if (first_event) {
    long pos=std::ftell(input);if (pos<4) std::abort();
    uint32_t meta[]={lo,hi,uint32_t(pos-4)};
    if (std::fwrite(meta,sizeof(meta),1,calibration_output)!=1) std::abort();
    first_event=false;
  }
  struct Event {uint32_t final,head;uint16_t prefix;uint8_t previous,bit;};
  static_assert(sizeof(Event)==12);
  Event event={p,head,uint16_t(prefix),calibration_previous,uint8_t(bit)};
  if (std::fwrite(&event,sizeof(event),1,calibration_output)!=1) std::abort();
}
"""
file.write_text(header.replace("BLOCK_IDS",",".join(map(str,GROUPS)))+code)
