# Reproducing the baseline and testing quantized weights

The baseline is `marwoelk/cmix-neif-2.0` at `35d1957a5074db802fc134126afc77b76d931761`. Its published score is 93,388,956 bytes. The rules and accounting are in [prize-rules.md](prize-rules.md). The experiment instructions are in [quant/program.md](quant/program.md).

## Baseline

The manual **Rebuild published compressor** workflow runs upstream’s install and offline build recipe inside Ubuntu 24.04 and verifies the resulting executable against the published SHA-256. On a native x86-64-v3 Linux worker, pass that binary or the hash-identical published executable and the original corpus to the driver. Pin its entire process tree to one allowed logical CPU.

```bash
taskset -c "$CPU" bash research/baseline.sh /path/to/comp9 /path/to/enwik9 /scratch/baseline-01
```

The driver checks both inputs, encodes, verifies the archive, starts that archive in a fresh directory, compares the decoded bytes and writes a score. The phase is in `status`. The timing reports use GNU time. This is a reproducibility run. Qualify a submission separately using the resource accounting and isolation described in the rules document.

## Fast benchmark

Use Linux x86-64 with AVX2/FMA, clang++-18 and lld-18 for the build, Python 3 and NumPy for fixture preparation and candidate export. The fork's **Build quantization tools** workflow exports the four native programs below as a tarball. To build them locally and prepare the initial suite, use absolute paths outside the checkout for scratch.

```bash
research/quant/build.sh /scratch/tools/replay
bash training/tf/1_dump/build_dump_tools.sh "$PWD" /scratch/tools bed dump priors
bash research/quant/prepare.sh /scratch/tools /data/enwik9 /scratch/c40 40000000
```

The one-time C40 preparation preprocesses the original corpus into the codec’s canonical stream, cuts it at upstream's 39,998,324-byte page boundary, and runs the native dump and PPMd tools. It needs 35 GB free scratch. Passing `full` in a separate scratch directory uses the complete stream and checks for 275 GB free. The retained replay suite is much smaller than the intermediate prior raster. The fixture manifests record document and stream offsets, scored targets, lengths and SHA-256s.

Each of the twelve windows starts at a real document reset, includes prompt rows and true float16 PPMd priors, and warms the first 256 fed rows before counting up to 4,096 more. Document-ending predictions are omitted. Metadata where the transformer is gated contributes no neural prediction gain. C40 samples the early corpus only. The full selection extends across late articles. Sampling from a reset reconstructs the primary head’s document state. The outer mixers, second head and semantic cache participate in full-codec validation.

```bash
python3 research/quant/evaluate.py /scratch/tools/replay /scratch/c40/suite \
  weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5 /scratch/results --label identity
python3 research/quant/loop.py /scratch/tools/replay /scratch/c40/suite \
  /scratch/results --fractions 0.125 0.25 0.5
```

The identity control must return zero on every window. A run records actual packed bytes and primary-head negative log likelihood. The proxy uses twice the container-size difference plus the mean window delta in bits per scored target multiplied by 487,298,491 upstream scored targets and divided by eight. This projection assumes the window estimate applies across the stream and that the neural delta transfers to the mixed codec. Calibrate those assumptions with full-codec trials before using it for a winning-margin estimate. Actual prize accounting uses the final artifact lengths and never adds a separate weight penalty.

`evaluate.py` prints `METRIC search_proxy_bytes=...` and related metrics for a command-based autoresearch controller. `loop.py` starts with a bounded three-point sweep. It snapshots each generator, verifies its exported tensors against the upstream NP2 decoder, and records failures. Its trial journal keys the cached reference by the fixture manifest, executable, evaluator and reference weights. Supply trusted, fixed-geometry exports. The release reader specializes tensor layout and the proxy reader verifies the chosen candidate’s fingerprint.

## Evidence from this fork

The [rANS comparison](ans/README.md) tests an alternative entropy coder against the release arithmetic coder on the same published bitstream.

The [baseline build](https://github.com/thomas-lanning/cmix-neif-2.0/actions/runs/44198808592) reproduced the published compressor hash. The [native research tools build](https://github.com/thomas-lanning/cmix-neif-2.0/actions/runs/44199917904) produced the replay and three preparation tools.
