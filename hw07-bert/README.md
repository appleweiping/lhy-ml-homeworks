# HW7 — BERT (Extractive Question Answering)

Fine-tune a real pretrained **BERT** for extractive QA: given a (question,
paragraph) pair, predict the answer span's start/end token positions — exactly
the official HW7 task. The official data is a Chinese reading-comprehension set
(DRCD/ODSQA); here we build an English extractive-QA dataset with the same
structure and use **`prajjwal1/bert-tiny`** (a real pretrained BERT small enough
for CPU).

To keep the task genuinely non-trivial, each context contains **three different
people**, each with their own city / job / year, and the question targets one of
them — so the model must locate the correct span **among distractors**, not do a
single-fact lookup. Train and validation use **disjoint entity pools**, so the
reported score measures real generalisation.

## Run
```bash
python hw7_bert_qa.py --epochs 8 --model prajjwal1/bert-tiny
```

## Measured result (CPU, 3 threads, multi-entity extractive QA)
| metric | value |
|---|---|
| validation Exact-Match | **0.9033** |
| validation token-F1 | **0.9033** |

bert-tiny learns to attend to the queried entity and extract the right span from
contexts with distractor entities it has not seen at train time. Sample
predictions in `results/samples.txt`.
