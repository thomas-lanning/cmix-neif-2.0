"""Install a 48-byte semantic key and a 25% larger ring within the old memory budget."""
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ans"))
from install import replace


def install(root):
    geom=root/"src/semantic_cache.h"
    s=replace(geom.read_text(),"constexpr uint32_t kN = 1u << kNLog;",
              "constexpr uint32_t kN = 20u << 20;")
    s=replace(s,"constexpr int kKB = kD / 2;", "constexpr int kKB = 3*kD / 8;")
    s=replace(s,"kKB * 2 == kD","kKB * 8 == kD * 3")
    geom.write_text(s)
    p=root/"src/predictor.cpp";s=p.read_text()
    s=replace(s,"    SemPut4(s.key + (size_t)ri * kKB, s.ck);","    cache_int3::pack(s.key + (size_t)ri * kKB, s.ck);")
    for old,new in (("s.nins & (kN - 1)","s.nins % kN"),("e & (kN - 1)","e % kN"),("pid[a] & (kN - 1)","pid[a] % kN")):
        s=replace(s,old,new)
    a=s.index("  // int4 cosine, exact in integers:")
    b=s.index("  for (int a = 0; a < npre;",a)
    s=s[:a]+s[b:]
    a=s.index("    __m256i da =",a)
    b=s.index("    const double sim =",a)
    s=s[:a]+"    int32_t d; uint32_t kn2;\n    cache_int3::dot(kk,s.ck,d,kn2);\n"+s[b:]
    # The count distinguishes a real expanded-ring test from the early cold start.
    s=replace(s,"    ++s.nins;","""    ++s.nins;
#ifdef NEIF_REPORT_CACHE_FILL
    if (s.nins % (1u<<18) == 0) std::fprintf(stderr,"cache_ring_entries %llu\\n",(unsigned long long)s.nins);
#endif""")
    p.write_text('#include "research_cache_int3.h"\n'+s)
    shutil.copyfile(HERE/"cache_int3.h",root/"src/research_cache_int3.h")


if __name__=="__main__":
    import subprocess
    root=Path(sys.argv[1]).resolve()
    if "--bounded" in sys.argv[2:]:
        subprocess.run([sys.executable,HERE/"instrument_prefix.py",root,"--no-events"],check=True)
        f=root/"src/runner.cpp"
        s=replace(f.read_text(),"  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);",
                  "  census_block_active=true;\n  isbn_check::after_at.an=true;\n  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);")
        s=replace(s,"    putc(byte, os);\n    sem_pb","    putc(byte, os);\n    if (pos==19999999) std::exit(fflush(os) ? 4 : 0);\n    sem_pb")
        f.write_text(s)
    install(root)
    if "--bounded" in sys.argv[2:]:
        f=root/"src/predictor.cpp";f.write_text("#define NEIF_REPORT_CACHE_FILL\n"+f.read_text())
