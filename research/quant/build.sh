#!/usr/bin/env bash
# Build against the fixed-geometry inference kernels from this checkout.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "${1:?output executable path}")
mkdir -p "$(dirname "$out")"
cd "$root"
clang++-18 -O3 -std=c++17 -march=x86-64-v3 -mtune=znver2 -mrecip=none \
  -ffp-contract=off -ffp-model=precise -fno-math-errno -fno-pie -no-pie \
  -ffunction-sections -fdata-sections -Wl,--gc-sections -DNEIF_QUANT_REPLAY \
  research/quant/replay.cpp src/tf1/kernels.cpp src/tf1/weights_io{,_compressed}.cpp \
  src/tf1/opt/*.cpp src/detmath.cpp -o "$out"
