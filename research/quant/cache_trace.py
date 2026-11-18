"""Capture matched cache queries and original eight-bit keys in a scratch encoder."""
from pathlib import Path
import shutil
import sys
from cache_keys import install


def once(text,anchor,extra):
    if text.count(anchor)!=1:
        raise ValueError(f"cache source differs at {anchor[:40]}")
    return text.replace(anchor,extra)


root=Path(sys.argv[1])
if sys.argv[2:]==["--decode-archive"]:
    runner=root/"src/runner.cpp"
    source=once(runner.read_text(),"  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);",
        "  census_block_active = true;\n  isbn_check::after_at.an = true;\n"
        "  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);")
    source=once(source,"    putc(byte, os);\n    sem_pb",
        "    putc(byte, os);\n"
        "    if (pos == 18000000 || pos == 30000000 || pos == 40000000) {\n"
        "      if (fflush(os)) std::exit(4);\n"
        "      if (pos == 40000000) std::exit(0);\n"
        "    }\n    sem_pb")
    runner.write_text(source)
elif len(sys.argv)==2:
    install(root,"sweep")
else:
    raise SystemExit("expected FRESH_TREE [--decode-archive]")
p=root/"src/predictor.cpp"
s=once(p.read_text(),"  TreffByteStart(strom_pb);",
       "  cache_trace::target(uint8_t(strom_pb),s.spos);\n  TreffByteStart(strom_pb);")
s=once(s,"    SemPut4(s.key + (size_t)ri * kKB, s.ck);",
       "    cache_trace::store(ri,s.ck);\n    SemPut4(s.key + (size_t)ri * kKB, s.ck);")
s=once(s,"  // 7. followers:","  cache_trace::sample(spos,s,pid,npre,kid,ks,nk,t);\n  // 7. followers:")
anchor='__attribute__((noinline)) void Predictor::SemByteStart(int strom_pb) {'
s=once(s,anchor,'#include "research_cache_trace.h"\n\n'+anchor)
p.write_text(s)
shutil.copyfile(Path(__file__).with_name("cache_trace.h"),root/"src/research_cache_trace.h")
