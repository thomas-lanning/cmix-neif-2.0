#!/usr/bin/env bash
# Run a reference encode followed by a decode of that freshly generated archive.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "usage: baseline.sh /path/to/comp9 /path/to/enwik9 new-run-directory" >&2
  exit 2
fi
comp=$(realpath "$1")
corpus=$(realpath "$2")
run=$(realpath -m "$3")
mkdir "$run"   # Refuse to mix results from two attempts.
cd "$run"
mkdir encode decode
phase() { echo "$1" > status.tmp && mv status.tmp status && echo "$1"; }
trap 'phase "FAILED line=$LINENO"' ERR
verify() { printf '%s  %s\n' "$1" "$2" | sha256sum -c -; }
phase verifying_inputs
verify 976bf2d309707ab555e0c92720da5508ff0e5e6b846fc7082234262b06767f82 "$comp"
verify 159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc "$corpus"
cp "$comp" encode/comp9
chmod 755 encode/comp9
cp --reflink=auto "$corpus" encode/enwik9
uname -a > host.txt
lscpu >> host.txt
taskset -pc $$ >> host.txt
phase encoding
(cd encode && /usr/bin/time -v -o ../encode.time ./comp9 >../encode.log 2>&1)
verify 88dd7087404122d010261e6855afd88875d34dde1f0539377d3e832fc0f6b3fd encode/archive9
cp encode/archive9 decode/archive9
chmod 755 decode/archive9
phase decoding
(cd decode && /usr/bin/time -v -o ../decode.time ./archive9 >../decode.log 2>&1)
cmp -s "$corpus" decode/enwik9_uncompressed
verify 159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc decode/enwik9_uncompressed
python3 - <<'PY'
from pathlib import Path
import json
c, a = [Path(p).stat().st_size for p in ("encode/comp9", "encode/archive9")]
assert (c, a) == (4042564, 89346392)
Path("result.json").write_text(json.dumps(dict(compressor=c, archive=a, arguments=0, score=c+a, roundtrip=True), indent=2) + "\n")
PY
phase reproduced
