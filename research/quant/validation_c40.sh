#!/usr/bin/env bash
# Rebuild the independent 48-article suite from a completed low-disk C40 preparation.
set -euo pipefail
tools=$(realpath "${1:?native tools directory}")
first=$(realpath "${2:?completed C40 working directory}")
out=$(realpath -m "${3:?fresh destination}")
code=$(cd "$(dirname "$0")" && pwd)
mkdir "$out"
echo "77974c6afc23485b60b0a8444b3060e465fcc9ae5b65cf1c75c5a0090b29753b  $first/suite/manifest.json" | sha256sum -c -
selection=(--windows 48 --exclude-manifest "$first/suite/manifest.json")
python3 "$code/fixtures.py" "$first" "$out/ranges.json" 40000000 "${selection[@]}" --plan-only
plan="$out/ranges.json"
if [[ ${4:-} == --training ]]; then
  # Exclude the original quick screen, all 48 validation documents and
  # the 24 articles used to calibrate the quick score to true payload bytes.
  excluded=(--exclude-manifest "$first/suite/manifest.json" --exclude-manifest "$plan")
  python3 "$code/fixtures.py" "$first" "$out/interval.json" 40000000 \
    --windows 24 --start 15282114 --stop 20000006 "${excluded[@]}" --plan-only
  training=(--windows 96 "${excluded[@]}" --exclude-manifest "$out/interval.json")
  python3 "$code/fixtures.py" "$first" "$out/training.json" 40000000 "${training[@]}" --plan-only
  python3 - "$plan" "$out/training.json" "$out/union.json" <<'PY'
import json, sys
from pathlib import Path
a,b = (json.loads(Path(p).read_bytes()) for p in sys.argv[1:3])
assert (a["prior_rows"],a["raw_sha256"]) == (b["prior_rows"],b["raw_sha256"])
spans = []
for lo,hi in sorted(a["spans"]+b["spans"]):
    if spans and lo <= spans[-1][1]:
        spans[-1][1] = max(hi,spans[-1][1])
    else:
        spans.append([lo,hi])
Path(sys.argv[3]).write_text(json.dumps(dict(prior_rows=a["prior_rows"],raw_sha256=a["raw_sha256"],spans=spans)))
PY
  plan="$out/union.json"
elif [[ $# -gt 3 ]]; then
  echo 'optional fourth argument is --training' >&2; exit 2
fi
cp "$first/bed/.dict" "$out/.dict"
cp "$first/bed/censpruef_vs3.bin" "$out/.vs3"
ln -s /dev/null "$out/aus"
(
  cd "$out"
  export OMP_NUM_THREADS=1 NEIF_TF1_WEIGHTS=none
  export NEIF_PPMD_ONLY_BYTES="$out/bytes.u8" NEIF_PPMD_ONLY_QUELLE="$out/origins.u8"
  python3 "$code/compact_priors.py" --plan "$plan" "$out/selected.float16" \
    "$tools/cmix_priors" -n "$first/c40" ./aus >priors.log 2>&1
)
cmp "$out/origins.u8" "$first/priors/ppmd-quelle.u8"
cmp "$out/bytes.u8" "$first/priors/ppmd-bytes.uint8"
python3 "$code/fixtures.py" "$first" "$out/suite" 40000000 "${selection[@]}" \
  --compact-priors "$out/selected.float16" --compact-plan "$plan"
echo "1f3c8d35380c9762babc40a11710098fc979e5eb37f0d21078ae86e1c3a85713  $out/suite/manifest.json" | sha256sum -c -
tar -C "$out" -czf "$out/c40-validation.tar.gz" ./suite
if [[ ${4:-} == --training ]]; then
  python3 "$code/fixtures.py" "$first" "$out/training" 40000000 "${training[@]}" \
    --compact-priors "$out/selected.float16" --compact-plan "$plan"
  tar -C "$out" -czf "$out/c40-training.tar.gz" ./training
fi
