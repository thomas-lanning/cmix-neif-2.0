# Quantization research loop

Read [the prize rules](../prize-rules.md) and the [operating instructions](../README.md). The objective is to minimize full packaged bytes subject to exact decoding and both resource limits. During screening use the six discovery windows and the doubled packed-weight cost. The other six are a confirmation set. Once used for choosing ideas, they are no longer an independent check.

1. Start from the pinned upstream weight file. Write one hypothesis and its predicted direction in your experiment branch. Edit `candidate.py` or an experimental training/export recipe. Preserve the published NP2 geometry for this replay.
2. Run the bounded sweep on the prepared native worker. Record the source hash, exported container hash, exact size, per-window effects and both proxy totals. The controller keeps the successfully exported model when both halves improve against upstream. Read previous trials before proposing another.
3. Compare promising exports against the current best on the same suite. Retain each model's hash and logs. Investigate disagreements across windows before combining hypotheses. Budget at most three exports per sweep. A process failure is a failed experiment.
4. For promotion, freeze the candidate, run a disjoint full-stream suite and then embed it in both halves of a release build. Restamp both weight checks against the actual bytes. Compression and decoding must agree on a clean x86 host. Score both executables and required arguments. Only a complete run and the resource checks establish a prize result.

The evaluator, fixture generator, scoring formula, baseline weights and accounting stay fixed during a sweep. GPU training and changing model dimensions are separate campaigns. A new geometry requires matching native kernels and a new calibration.
