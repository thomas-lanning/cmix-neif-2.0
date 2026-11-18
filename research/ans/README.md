# Binary rANS on the cmix decision stream

The release uses a binary arithmetic coder. Its predictor assigns the probability of the next bit from the preceding decoded symbols. This experiment tests Jarek Duda's range asymmetric numeral systems (rANS) on exactly those probabilities and symbols. The encoder buffers a bounded block of decisions in forward order and feeds them to rANS in reverse. The decoder consumes each block in forward order, so an integrated decoder could regenerate the same probabilities from its model. [Release encoder](../../src/coder/encoder.cpp) · [Duda's ANS paper](https://arxiv.org/abs/1311.2540)

## First measurement

The first 131,072 entropy-coded decisions from the hash-verified upstream archive took **13,692 bytes** with the release arithmetic algorithm, including one byte to terminate this prefix independently. The preceding 13,691 bytes matched the published payload byte for byte. All eight rANS runs recovered every decision, consumed their entire payload and reached the expected final state. Counts include four bytes of length and eight bytes of state per rANS block. [Workflow and downloadable trace](https://github.com/thomas-lanning/cmix-neif-2.0/actions/runs/44201406171)

| Decisions per block | Probability precision | rANS bytes | Difference from arithmetic |
| ---: | ---: | ---: | ---: |
| 4,096 | 12 bits | 14,060 | +368 |
| 4,096 | 16, 24 or 32 bits | 14,059 | +367 |
| 65,536 | 12, 16, 24 or 32 bits | 13,715 | +23 |
| 131,072 | 24 bits (single-block replay of the downloaded trace) | 13,704 | +12 |

The model's ideal length on these decisions, computed by summing the negative log probabilities, was 13,692.27 bytes. Encoding in a single block cost 12 bytes above arithmetic, exactly the size of the frame header and final state. Reproduce that row with `--precision 24 --block-size 131072`. The sample covers the start of the archive. A useful next comparison is a longer capture in the article body, measuring both compressed bytes and native single-core time. The Python replay measures byte counts and correctness. A full-codec trial also has to count changes in both executable sizes, regenerate probabilities in the decoder and round-trip `enwik9`.

## Reproduction

Run the fork's **Compare rANS against published arithmetic stream** workflow. It instruments a scratch copy of the release decoder, checks the archive hash, captures only decisions for which the codec consumed entropy and uploads `ans-prefix`. To repeat the byte comparison on that artifact, run

```sh
python3 research/ans/compare.py /path/to/coded-events.bin
```

The workflow additionally supplies `--original-payload` to verify the arithmetic bytes against the archive. The quantization suite uses the primary transformer's probabilities. Its results and this final mixed-bit trace answer different experimental questions.

A separate causal probability correction in `calibration.cpp` estimates the observed-to-expected error ratio by confidence bucket and recent coded bits. On the 131,072-decision prefix, selecting among 54 configurations using the first half gave a zero-byte change on the second half. The best whole-prefix change was zero and the worst cost 31 bytes. The arithmetic control independently reproduced the 13,692-byte length. Article text provides a separate test of this model change.

## Lossless rANS for the weights

A separate experiment encoded the signed weight symbols directly with one 15-symbol rANS distribution per tensor. It stores frequency tables and metadata, compresses the remaining byte planes with DEFLATE, and reconstructs the rotary tables with the upstream algorithm. Its 3,796,587-byte container recovered all 597 tensor names, types, shapes and values exactly. The shipped NP2 container used 3,744,629 bytes, giving NP2 a 51,958-byte advantage per copy. NP2 also encodes information in neuron permutations. The rANS research container's SHA-256 is `f4ec71872a32afa0181a9f39864111ef3c49ed17386f8cf36c64e1c2f87fac7b`. The measurement covers the stored data. A submission would also price its native reader.

```sh
python3 research/ans/weight_container.py repack weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5 /tmp/weights.ans
python3 research/ans/weight_container.py unpack /tmp/weights.ans /tmp/recovered.raw
```
