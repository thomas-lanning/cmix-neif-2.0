"""Install the adaptive rANS weight reader and its checked model in a fresh release tree."""
import argparse
import hashlib
from pathlib import Path
import shutil
import struct
from hybrid_weights import HEADER, MAGIC, unpack

WEIGHTS="weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5"
ORIGINAL="0xaec368ac6a568632ULL"


def once(text,before,after):
    if text.count(before)!=1:
        raise ValueError(f"upstream source differs at {before[:55]}")
    return text.replace(before,after,1)


def install(root,weights,sha,codec,reference):
    raw=weights.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha or raw[:8]!=MAGIC:
        raise ValueError("hybrid weight fingerprint or magic differs")
    _,count,_,n,m=HEADER.unpack_from(raw,8)
    cursor=8+HEADER.size+n
    end=cursor+m
    last=-1
    while cursor<end:
        i,size=struct.unpack_from("<HI",raw,cursor);cursor+=6
        if not last<i<count or size<6 or raw[cursor]!=0 or cursor+size>end:
            raise ValueError("the release reader accepts ordered adaptive rANS frames")
        last=i;cursor+=size
    if not 0<end<=len(raw) or end!=cursor or unpack(raw,codec)!=reference.read_bytes():
        raise ValueError("decoded hybrid differs from the reference NP2")
    fingerprint=14695981039346656037
    for byte in raw:
        fingerprint=(fingerprint^byte)*1099511628211 & 0xffffffffffffffff
    code=f"0x{fingerprint:016x}ULL"
    reader=root/"src/tf1/weights_io_compressed.cpp"
    text=once(reader.read_text(), '#include "weights_io.h"', '#include "weights_io.h"\n#include "residual_matrix.h"')
    text=once(text,"const uint32_t nA = r.u32(), lenA = r.u32(), K = r.u32();","""const uint32_t nA = r.u32(), K = r.u32(), lenA = r.u32(), extra = r.u32();
  r.need(size_t(lenA) + extra);
  Reader frames{r.p + lenA, extra, path};
  const size_t tail_offset = size_t(lenA) + extra;""")
    text=once(text,"(r.left - lenA)","(r.left - tail_offset)")
    text=once(text,"r.p + lenA, r.left - lenA","r.p + tail_offset, r.left - tail_offset")
    text=once(text,"  int decode_cbit(CountNode* n, uint32_t limit) {","""  static void observe(CountNode* n, unsigned bit, uint32_t limit) {
    if (bit) ++n->n1; else ++n->n0;
    if (uint32_t(n->n0) + n->n1 > limit) {
      n->n0 = std::max(1u, unsigned(n->n0) >> 1);
      n->n1 = std::max(1u, unsigned(n->n1) >> 1);
    }
  }
  int decode_cbit(CountNode* n, uint32_t limit) {""")
    text=once(text,"""    if (bit) n->n1++; else n->n0++;
    if (uint32_t(n->n0) + n->n1 > limit) {
      n->n0 = uint16_t(n->n0 >> 1 ? n->n0 >> 1 : 1);
      n->n1 = uint16_t(n->n1 >> 1 ? n->n1 >> 1 : 1);
    }""","    observe(n, bit, limit);")
    text=once(text,"        uint64_t* cs =","""        bool external = false;
        if (frames.left) {
          frames.need(6);
          uint16_t id;
          std::memcpy(&id, frames.p, 2);
          if (id < i) msg_stop(235);
          if (id == i) {
            frames.read(&id,2);
            uint32_t length = frames.u32();
            frames.need(length);
            if (length < 6 || frames.p[0] != 0 || frames.p[1] >= 32 || t.ndim != 2) msg_stop(235);
            matrix_ans::Reader input(frames.p + 2,length - 2);
            matrix_ans::selected_pass(out,rows,cols,frames.p[1],input);
            input.finish();
            frames.p += length;
            frames.left -= length;
            external = true;
          }
        }
        uint64_t* cs =""")
    text=once(text,"          if (dec.decode_cbit(&m->zrow, CNT_LIMIT_ZROW)) continue;","""          if (external) {
            bool zero = std::all_of(row,row+cols,[](int8_t value){return value == 0;});
            BinDecoder::observe(&m->zrow,zero,CNT_LIMIT_ZROW);
            if (zero) continue;
          } else if (dec.decode_cbit(&m->zrow, CNT_LIMIT_ZROW)) continue;""")
    text=once(text,"            int v = int(dec.decode_ctree(&m->cint4w[b * 16], 4, CNT_LIMIT_INT4W)) - 7;","""            CountNode* nodes = &m->cint4w[b * 16];
            int v;
            if (external) {
              v = row[k];
              unsigned node = 1;
              for (int shift = 3; shift >= 0; --shift) {
                unsigned bit = (unsigned(v + 7) >> shift) & 1u;
                BinDecoder::observe(&nodes[node],bit,CNT_LIMIT_INT4W);
                node = node*2 + bit;
              }
            } else v = int(dec.decode_ctree(nodes,4,CNT_LIMIT_INT4W)) - 7;""")
    text=once(text,"  if (n_cn != size_t(kNamenN)", "  if (frames.left) msg_stop(235);\n  if (n_cn != size_t(kNamenN)")
    text=once(text,ORIGINAL,code)
    extra={}
    for path,num in (("src/transformer.h",1),("src/tf1/weights_io.h",2)):
        p=root/path
        s=p.read_text()
        if s.count(ORIGINAL)!=num:
            raise ValueError(f"unexpected fingerprint in {path}")
        extra[p]=s.replace(ORIGINAL,code)
    for p,s in extra.items():p.write_text(s)
    reader.write_text(text)
    shutil.copyfile(Path(__file__).with_name("adaptive_matrix.h"),reader.parent/"adaptive_matrix.h")
    shutil.copyfile(Path(__file__).with_name("residual_matrix.h"),reader.parent/"residual_matrix.h")
    (root/WEIGHTS).write_bytes(raw)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path,help="fresh source tree at the upstream baseline")
    p.add_argument("weights",type=Path)
    p.add_argument("sha256")
    p.add_argument("matrix_tool",type=Path)
    p.add_argument("--reference",type=Path,help="underlying NP2 model, defaults to the released file in the tree")
    a=p.parse_args()
    install(a.root,a.weights,a.sha256,a.matrix_tool.resolve(),a.reference or a.root/WEIGHTS)
