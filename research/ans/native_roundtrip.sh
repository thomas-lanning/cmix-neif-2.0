#!/usr/bin/env bash
# Run the bounded rANS codec in independent encoding and decoding processes.
set -euo pipefail
if [[ $# != 5 || ( $5 != quick && $5 != body ) ]]; then
  echo 'usage: native_roundtrip.sh CMIX_RANS CANONICAL_STREAM PUBLISHED_ARCHIVE NEW_DIR quick|body' >&2
  exit 2
fi
repo=$(cd "$(dirname "$0")/../.." && pwd)
bin=$(realpath "$1")
canonical=$(realpath "$2")
archive=$(realpath "$3")
out=$(realpath -m "$4")
size=$5
weights=weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5
check() { printf '%s  %s\n' "$1" "$2" | sha256sum -c -; }
check c35d3073eeba70f22a2e08a50202b53eb95466e3e9f1f2702f6f9b3f96fd8718 "$canonical"
check 88dd7087404122d010261e6855afd88875d34dde1f0539377d3e832fc0f6b3fd "$archive"
check 729694cba4969b7d833686ed3f597ff04d3b226da849d5f8004ec020a0ed9563 "$repo/$weights"
mkdir "$out"
cd "$out"
mkdir encode decode
cp "$repo/$weights" encode/
cp "$repo/$weights" decode/
cp "$repo/dictionary/english.dic" encode/.dict
export NEIF_CANDIDATE_FNV=aec368ac6a568632
unset NEIF_CANONICAL_STREAM NEIF_RANS_REPLAY NEIF_QUICK_CHECK
length=20000000
if [[ $size == quick ]]; then export NEIF_QUICK_CHECK=1; length=200000; fi
(
  cd encode
  NEIF_CANONICAL_STREAM="$canonical" /usr/bin/time -v -o time "$bin" -e "$canonical" payload >native.log 2>&1
)
(
  cd decode
  NEIF_RANS_REPLAY=1 /usr/bin/time -v -o time "$bin" -d ../encode/payload output >native.log 2>&1
)
test "$(stat -c%s decode/output)" = "$length"
cmp -n "$length" decode/output "$canonical"
python3 - "$archive" "$length" "$bin" <<'PY'
import hashlib,json,pathlib,struct,sys
p=pathlib.Path
oracle=p("encode/arithmetic.oracle").read_bytes()
payload=p("encode/payload").read_bytes()
with open(sys.argv[1],"rb") as f:
    f.seek(-8,2)
    size,weight=struct.unpack("<II",f.read(8))
    if weight!=3744629:raise ValueError("unexpected release weight length")
    f.seek(-8-size,2)
    if len(oracle)<40000 or f.read(5)!=payload[:5] or f.read(len(oracle))!=oracle:
        raise ValueError("arithmetic oracle differs from the published archive")
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
out=dict(canonical_bytes=int(sys.argv[2]),fresh_decoder_roundtrip=True,native_tool_sha256=sha(p(sys.argv[3])),
         canonical_prefix_sha256=sha(p("decode/output")),common_header_bytes=5,
         original_arithmetic_bytes=5+len(oracle)+1,rans_bytes=len(payload),rans_sha256=hashlib.sha256(payload).hexdigest(),
         published_continuing_prefix_bytes=5+len(oracle))
p("result.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))
PY
