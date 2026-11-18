# Binary rANS on the cmix decision stream

The release uses a binary arithmetic coder. Its predictor assigns the probability of the next bit from the preceding decoded symbols. This experiment tests Jarek Duda's range asymmetric numeral systems (rANS) on exactly those probabilities and symbols. The encoder buffers a bounded block of decisions in forward order and feeds them to rANS in reverse. The decoder consumes each block in forward order, so an integrated decoder could regenerate the same probabilities from its model. [Release encoder](../../src/coder/encoder.cpp) · [Duda's ANS paper](https://arxiv.org/abs/1311.2540)

## First 131,072 coded decisions

The release-profiled encoder ran on the canonical 501,062,281-byte stream. Two builds, one with probability logging and one without, matched the first 42,693 emitted bytes against the published archive. The recorded trace has SHA-256 `95f04fea97d97423fae673febe8e79edce54d87973a3b4dd8ced3be8523a62ae`. Replaying the trace with arithmetic coding reproduced all 3,985 continuing bytes. Terminating this short sample independently gives **3,986 bytes**. The ideal length from the recorded probabilities is 3,985.22 bytes.

| Method | Probability precision | Stored bytes | Difference from arithmetic |
| --- | --- | ---: | ---: |
| Two rANS blocks of 65,536 decisions | 12 bits | 4,010 | +24 |
| Two rANS blocks of 65,536 decisions | 16, 24 or 32 bits | 4,008 | +22 |
| One length-prefixed rANS block | 12 bits | 3,998 | +12 |
| One length-prefixed rANS block | 16, 24 or 32 bits | 3,997 | +11 |
| One continuous rANS state | 16, 24 or 32 bits | **3,991** | **+5** |

Each decoder recovered every decision and consumed its entire payload. In the continuous implementation, one six-byte state serves the entire stream. It reads probabilities and outcomes backwards in 65,536-decision chunks while encoding and decodes in forward order. For this sample its on-disk forward trace used 655,360 bytes and its reverse output spool used 3,985 bytes. The outer format must supply a stopping condition. The release already carries the decoded stream length. A prize trial would still need an integrated encoder and decoder, executable-size accounting and a full resource-qualified round trip.

An earlier unprofiled decoder produced a 13,692-byte arithmetic result for a different sequence of decoded decisions. Re-encoding those decisions can reproduce the input bytes even when the model diverges. The benchmark now checks the independent encoding trace as well as arithmetic bytes. The release build settings and the public payload are part of the reference. [Build recipe](../../build_and_construct_comp.sh) · [Initial trace workflow](../../.github/workflows/ans.yml)

The [profile-guided CI reproduction](https://github.com/thomas-lanning/cmix-neif-2.0/actions/runs/44216165137) passed the independently captured decision digest.

## Native encode and independent decode

`install.py SOURCE --prefix-test` stages a research build in a fresh source tree. It inserts rANS after the existing predictor, including the special raw-bit path. Each block holds up to 262,144 coded decisions. The decoder counts the same decisions and reads each six-byte state directly, so block boundaries need no length field. It regenerates neural and context-mixer probabilities from decoded bytes. The input stream length supplies the stop condition. The encoder's research-only arithmetic output is checked against the published archive.

On 200,000 canonical input bytes, the rANS stream occupied **42,728 bytes**, compared with **42,694** for independently terminated arithmetic coding, including the same five-byte header. A fresh decoder process, given the rANS stream and the published weights, reproduced all 200,000 canonical bytes. It checked the terminal rANS state and consumed the complete input. All 42,688 continuing arithmetic-coded bytes after the header matched the published archive. The small sample ends in the metadata region.

From the fork root on x86-64 Linux, build in a fresh scratch directory. The benchmark preparation produces the full canonical stream in `bed/censpruef_strom.bin`. The driver verifies that stream, the weights and the published self-extracting archive by SHA-256.

```bash
mkdir /scratch/cmix-rans
git archive HEAD src makefile encoder_split.lds runtime_exports.map pgo_data | tar -x -C /scratch/cmix-rans
python3 research/ans/install.py /scratch/cmix-rans --prefix-test
make -C /scratch/cmix-rans -j2 prof_use CC=clang++-18
bash research/ans/native_roundtrip.sh /scratch/cmix-rans/cmix /scratch/full/bed/censpruef_strom.bin \
  /data/published/archive9 /scratch/rans-200k quick
```

Choose `body` with a different output directory to encode and decode the first 20 million canonical bytes. The result file records both stored lengths, the fresh decoder's SHA-256 and the matched arithmetic prefix.

The separate causal correction in `calibration.cpp` estimates the observed-to-expected error ratio from confidence and previous coded bits. Selecting among 54 settings on the first half of the verified trace saved four bytes on that half and two on the second. Actual packaged code costs would also count.

## Reproduction

The fork's **Compare rANS against published arithmetic stream** workflow builds the decoder with upstream's profile and checks the recorded probabilities and bits against the verified encoder trace hash. To test a downloaded trace locally, run

```sh
python3 research/ans/compare.py /path/to/coded-events.bin --precision 12 16 24 32 --block-size 65536 262144
c++ -O3 -std=c++17 research/ans/continuous.cpp -o /tmp/neif-continuous
/tmp/neif-continuous /path/to/coded-events.bin /tmp/sample.rans 24
```

To time the coder alone on an identical decision sequence, compile `research/ans/coder_speed.cpp` with `c++ -O3 -std=c++17` and pass the trace filename. The program alternates arithmetic and block rANS, checks every decoded decision, discards a warm-up pair and reports the median and median absolute deviation of nine measurements. Native whole-model timing includes the predictor and must be measured separately.

## Lossless rANS for the weights

A separate experiment encoded the signed weight symbols directly with one 15-symbol rANS distribution per tensor. It stores frequency tables and metadata, compresses the remaining byte planes with DEFLATE, and reconstructs the rotary tables with the upstream algorithm. Its 3,796,587-byte container recovered all 597 tensor names, types, shapes and values exactly. The shipped NP2 container used 3,744,629 bytes, giving NP2 a 51,958-byte advantage per copy. NP2 also encodes information in neuron permutations. The rANS research container's SHA-256 is `f4ec71872a32afa0181a9f39864111ef3c49ed17386f8cf36c64e1c2f87fac7b`. The measurement covers the stored data. A submission would also price its native reader.

```sh
python3 research/ans/weight_container.py repack weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5 /tmp/weights.ans
python3 research/ans/weight_container.py unpack /tmp/weights.ans /tmp/recovered.raw
```

### Mixing arithmetic and adaptive rANS inside the weight container

A second experiment preserves the 20,349 bytes carried by NP2's neuron permutations. It uses rANS for selected signed integer matrices and continues to update NP2's arithmetic model with their true values. This keeps the original encoding context for the following tensors. For each selected matrix, the adaptive rANS reader derives symbol probabilities from previously recovered weights in the same row and column. The container stores the chosen traversal and model variant. All selection and framing bytes count.

With 62 matrices selected, the complete weight file measured **3,739,290 bytes**, saving **5,339 bytes per embedded copy**. Its SHA-256 is `a13dd39b10b9bb1439acc3c084558599943810c9066ea946afb336c49dcf60a4`. The Python unpacker regenerated the entire published NP2 file byte for byte. Independent native readers for NP2 and the hybrid produced identical bytes for all 597 tensor slots, shapes, types and values. The native dump was 43,058,716 bytes with SHA-256 `5bb41b9479506c8906b4f5e48dc262270bca170421937182752693405c9ac6ba`. The executable size of the new reader also counts in a full entry.

```bash
c++ -O3 -std=c++17 research/ans/adaptive_matrix.cpp -o /tmp/neif-matrix
python3 research/ans/hybrid_weights.py pack weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5 \
  /tmp/weights.hybrid --matrix-model adaptive --matrix-tool /tmp/neif-matrix
python3 research/ans/hybrid_weights.py unpack /tmp/weights.hybrid /tmp/recovered.c5 --matrix-tool /tmp/neif-matrix
cmp /tmp/recovered.c5 weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5
```

To build the candidate, extract upstream commit `35d1957a` into a fresh directory and run `stage_hybrid.py FRESH_TREE HYBRID_FILE HYBRID_SHA256 MATRIX_TOOL`. Build that tree with the published release toolchain. The primary payload keeps the shipped binary arithmetic coder.

The release build produced a **4,038,356-byte compressor**. Its packed decoder is 185,018 bytes. Combining that decoder, the new weights and the published arithmetic payload produced an 89,342,186-byte self-extracting archive. This gives **93,380,542 bytes** if the fresh encode reproduces the complete published payload, a net saving of **8,414 bytes** including the new code. Reassembly from the original compressor reproduced the published archive SHA-256 exactly. A fresh candidate encode and an independent decode of the reassembled archive are running. [Package measurements and digests](hybrid-result.json)
