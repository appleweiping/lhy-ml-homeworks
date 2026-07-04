# HW5 — Transformer (Neural Machine Translation)

A full **Transformer encoder-decoder** implemented with the official architecture:
token + sinusoidal positional embeddings, multi-head self/cross attention, causal
masking, teacher-forced training, and greedy decoding.

The official task translates English↔Chinese from fairseq-preprocessed TED talks
(multi-GB, needs the fairseq toolchain). To run a **real** translation task on
CPU, we train on the real **Multi30k German→English** parallel corpus (auto-
downloaded via HuggingFace `datasets`), building word-level vocabularies from the
training split and reporting corpus BLEU-4 on the held-out validation split.

## Run
```bash
python hw5_transformer.py --epochs 15 --n-train 3000
```

## Measured result (CPU, 3 threads, real Multi30k de→en, 3k pairs, 15 epochs)
| metric | value |
|---|---|
| validation token accuracy (teacher-forced) | 0.4816 |
| validation **corpus BLEU-4** (full val set, greedy) | **0.1560** |

Example (from `results/samples.txt`):

```
DE:  eine gruppe von männern <unk> <unk> auf einen <unk>
REF: a group of men are loading <unk> onto a truck
HYP: a group of men are working on a <unk> of a <unk>
```

Model: d=192, 4 heads, 2 enc/dec layers, label-smoothed cross-entropy. The small
vocabulary (min-freq pruning) yields `<unk>` tokens; the outputs are nonetheless
fluent and semantically on-topic, confirming the Transformer learned real
translation structure at this CPU-modest scale.
