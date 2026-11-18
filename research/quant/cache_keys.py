"""Stage a semantic-cache key quantization experiment in a fresh release tree."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def install(root, scheme):
    here=Path(__file__).resolve().parent
    if scheme=="sweep":
        subprocess.run([sys.executable,here/"instrument_prefix.py",root,"--no-events"],check=True)
        runner=root/"src/runner.cpp"
        text=runner.read_text()
        text=text.replace("200000 : 20000000;","200000 : 42000000;")
        text=text.replace("pos == 20000000)","pos == 20000000 || pos == 25000000 || pos == 30000000 || pos == 35000000 || pos == 40000000 || pos == 42000000)")
        if text.count("42000000")!=2:raise ValueError("prefix stop or checkpoints changed")
        runner.write_text(text)
    path=root/"src/predictor.cpp"
    anchor="static inline void SemPut4(uint8_t* dst, const int8_t* k) {"
    twin='  research_cache::pack(dst, k, '+("true" if scheme=="angle" else "false")+');\n  return;'
    if scheme=="sweep":
        twin="""
  static const int mode = [] {
    const char* m = std::getenv("NEIF_CACHE_SCHEME");
    if (!m || !std::strcmp(m,"original")) return 0;
    if (!std::strcmp(m,"clip")) return 1;
    if (!std::strcmp(m,"angle")) return 2;
    std::abort();
  }();
  if (mode) { research_cache::pack(dst,k,mode==2); return; }
"""
    text=path.read_text()
    if text.count(anchor)!=1:raise ValueError("release cache quantization changed")
    shutil.copyfile(here/"cache_keys.h",root/"src/cache_keys_experiment.h")
    path.write_text('#include "cache_keys_experiment.h"\n'+text.replace(anchor,anchor+"\n"+twin))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("scheme",choices=("sweep","clip","angle"))
    a=p.parse_args();install(a.source.resolve(),a.scheme)
