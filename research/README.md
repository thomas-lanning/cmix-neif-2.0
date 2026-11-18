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

The [baseline build](https://github.com/thomas-lanning/cmix-neif-2.0/actions/runs/44198808592) reproduced the published compressor hash. A [second release build with the research code present](https://github.com/thomas-lanning/cmix-neif-2.0/actions/runs/44201814970) reproduced it again. The [native research tools build](https://github.com/thomas-lanning/cmix-neif-2.0/actions/runs/44199917904) produced the replay and three preparation tools.

The initial C40 suite contains 42,238 fed rows and 34,706 scored targets across twelve article windows. One native pass through all windows took 15.53 seconds after the reference was cached. Timing excludes one-time fixture generation and candidate export. Replaying the original file returned zero on every window. A separate export with zero changed rows reproduced all 3,744,629 published weight bytes.

| Final MLP rows coarsened | Packed weights saved | Discovery proxy Δ | Confirmation proxy Δ |
| ---: | ---: | ---: | ---: |
| 12.5% | 305 B | -6,346 B | +13,867 B |
| 25% | 683 B | -10,396 B | +14,916 B |
| 50% | 1,656 B | -12,065 B | +12,599 B |

Negative proxy values predict a smaller entry. The full-codec effects are unmeasured. The [pilot data](quant/c40-pilot.json) include the exact candidate and suite hashes, model losses and per-window deltas.

The first representation that preserved the prediction on every tested window flips the signs of selected integer weight rows and their BF16 scales together. The exporter verifies every effective weight and the neuron order after unpacking. The frozen container is 3,744,283 bytes, a saving of 346 bytes. Prediction loss matched on the initial twelve articles and all 48 additional early articles. A release build with this container produced a 4,042,218-byte compressor with SHA-256 `90dac9bbbafd78c29ae31109d2725472203efeb93881651d154451a925066062`. Its archive must be measured independently.

```bash
python3 research/quant/symmetry.py export tables fit .06 /scratch/sign-fit.c5
sha256sum /scratch/sign-fit.c5
# 78ef4ef61a6848e8a3c924eca62c87fe2110586ae8e5323fa303ec4d40d42eaa
```

The fast benchmark uses a second group of 48 early articles, a 24-article suite matched to the actual archive interval, and a disjoint later-corpus suite for promotion. Offline fitting uses a separate 96-article suite. `research/quant/verify_priors.py c40 /scratch/c40` checks all seven C40 raster digests against the published eight-digit prefixes. Its `full` mode validates the complete trimmed priors and token streams against the upstream SHA-256 values.
