#!/usr/bin/env bash
# Generate upstream's C40 raw rows and priors in private scratch, then select a compact suite.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
tools=$(realpath "${1:?directory containing cmix_bed, cmix_dump and cmix_priors}")
input=$(realpath "${2:?enwik9 file}")
work=$(realpath -m "${3:?new scratch directory}")
prefix=${4:-40000000}
[[ $prefix == 40000000 || $prefix == full ]] || { echo 'choose 40000000 or full' >&2; exit 2; }
need=35000000000
[[ $prefix != full ]] || need=275000000000
collector=()
fixture_args=()
if [[ ${5:-} == --compact && $prefix == 40000000 ]]; then
  need=8000000000
  collector=(python3 "$root/research/quant/compact_priors.py" "$work/priors/c40-selected.float16")
  fixture_args=(--compact-priors "$work/priors/c40-selected.float16")
elif [[ $# -gt 4 ]]; then
  echo '--compact supports the twelve-window C40 suite' >&2; exit 2
fi
mkdir "$work"
cd "$work"
[[ $(df -PB1 . | awk 'END {print $4}') -gt $need ]] || { echo 'insufficient scratch space' >&2; exit 2; }
printf '%s  %s\n' 159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc "$input" | sha256sum -c -
mkdir bed dump priors
cp "$root/dictionary/english.dic" bed/.dict
cp "$root/src/readalike_prepr/data/new_article_order" bed/.new_article_order
(cd bed && "$tools/cmix_bed" -e "$input" out >bed.log 2>&1)
if [[ $prefix == full ]]; then
  bed="$work/bed/censpruef_kanon.bin"
else
  # Published C40 ends at a complete page anchor.
  head -c 39998324 bed/censpruef_kanon.bin > c40
  bed="$work/c40"
fi
export OMP_NUM_THREADS=1 NEIF_TF1_WEIGHTS=none
for stage in dump priors; do
  cp bed/.dict "$stage/.dict"
  cp bed/censpruef_vs3.bin "$stage/.vs3"
  ln -s /dev/null "$stage/aus"
done
(cd dump && NEIF_TF1_PAKET_DUMP=./ NEIF_TF1_GATE_DUMP="$PWD/zonenmaske.u8" \
 NEIF_TF1_MATCH_DUMP=./ NEIF_TF1_SAVE_PRIORS=/dev/null \
 "$tools/cmix_dump" -n "$bed" ./aus >dump.log 2>&1) &
dump_pid=$!
set +e
(cd priors && NEIF_TF1_SAVE_PRIORS="$PWD/ppmd-probs.float16" \
 NEIF_PPMD_ONLY_BYTES="$PWD/ppmd-bytes.uint8" NEIF_PPMD_ONLY_QUELLE="$PWD/ppmd-quelle.u8" \
 "${collector[@]}" "$tools/cmix_priors" -n "$bed" ./aus >priors.log 2>&1)
prior_rc=$?
wait "$dump_pid"
dump_rc=$?
set -e
[[ $prior_rc == 0 && $dump_rc == 0 ]] || { echo "priors=$prior_rc dump=$dump_rc" >&2; exit 1; }
python3 "$root/research/quant/fixtures.py" "$work" "$work/suite" "$prefix" "${fixture_args[@]}"
if (( ${#collector[@]} )); then
  echo '77974c6afc23485b60b0a8444b3060e465fcc9ae5b65cf1c75c5a0090b29753b  suite/manifest.json' | sha256sum -c -
fi
