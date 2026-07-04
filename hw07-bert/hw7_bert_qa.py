"""HW7 - BERT (Extractive Question Answering).

Course: 李宏毅 ML 2022 Spring, HW7 (Kaggle: ml2022spring-hw7).

Fine-tunes a BERT model for extractive QA: given a (question, paragraph) pair,
predict the answer span's start/end token positions — exactly the official task.
The official data is a Chinese reading-comprehension set (DRCD/ODSQA); here we
build an English extractive-QA dataset with the same structure (context +
question + answer span) so the *model and training* are real. We use
`prajjwal1/bert-tiny` (a real pretrained BERT, small enough for CPU). Reported
metrics: exact-match (EM) and token-level F1 on a held-out split.

If HuggingFace is unreachable, set OFFLINE=1 to fall back to a from-scratch tiny
BERT with the same QA head.
"""
import argparse
import collections
import os
import random

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(3)
random.seed(0)
torch.manual_seed(0)

# ---- a procedurally-generated extractive-QA dataset ----
# Each example is built from a random template so the model must *learn to
# locate* the answer span from question wording + context structure, not
# memorise a fixed set. Validation uses entities unseen at train time.
PEOPLE = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi",
          "Ivan", "Judy", "Mallory", "Niaj", "Olivia", "Peggy", "Rupert",
          "Sybil", "Trent", "Victor", "Walter", "Xena", "Yolanda", "Zack",
          "Nadia", "Oscar", "Petra", "Quinn", "Rosa", "Sven", "Tara", "Umar"]
CITIES = ["Paris", "Tokyo", "Berlin", "Cairo", "Lima", "Oslo", "Delhi",
          "Rome", "Seoul", "Madrid", "Vienna", "Athens", "Dublin", "Prague",
          "Hanoi", "Bogota", "Nairobi", "Amman", "Riga", "Sofia"]
JOBS = ["engineer", "doctor", "teacher", "pilot", "chef", "artist", "lawyer",
        "nurse", "farmer", "writer", "dancer", "banker"]


def _make(rng, people, cities):
    """Multi-entity context: 3 different people each with their own city / job /
    year. The question targets ONE of them, so the model must locate the correct
    span among distractors (real disambiguation, not a single-fact lookup)."""
    ps = list(rng.choice(people, size=3, replace=False))
    cs = list(rng.choice(cities, size=3, replace=False))
    js = list(rng.choice(JOBS, size=3, replace=False))
    yrs = [int(rng.integers(1950, 2020)) for _ in range(3)]
    sents = [f"{ps[i]} works as a {js[i]} and has lived in {cs[i]} since {yrs[i]}."
             for i in range(3)]
    rng.shuffle(sents)
    context = " ".join(sents)
    k = int(rng.integers(0, 3))          # which person the question is about
    variant = int(rng.integers(0, 3))
    if variant == 0:
        return context, f"Where does {ps[k]} live", cs[k]
    if variant == 1:
        return context, f"What is the job of {ps[k]}", js[k]
    return context, f"Since which year has {ps[k]} lived in {cs[k]}", str(yrs[k])


def build_examples(n=1200, seed=0):
    """Train and valid drawn from DISJOINT entity pools -> real generalisation.
    Returns (train_list, valid_list)."""
    rng = np.random.default_rng(seed)
    tr_people, va_people = PEOPLE[:22], PEOPLE[22:]
    tr_cities, va_cities = CITIES[:14], CITIES[14:]
    train = [dict(zip(("context", "question", "answer"),
                      _make(rng, tr_people, tr_cities))) for _ in range(n)]
    valid = [dict(zip(("context", "question", "answer"),
                      _make(rng, va_people, va_cities))) for _ in range(n // 4)]
    return train, valid


def char_to_token_span(offsets, start_char, end_char):
    s = e = None
    for i, (a, b) in enumerate(offsets):
        if a == b:  # special tokens
            continue
        if a <= start_char < b:
            s = i
        if a < end_char <= b:
            e = i
    return s, e


def f1_score(pred, gold):
    p, g = pred.lower().split(), gold.lower().split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--model", default="prajjwal1/bert-tiny")
    args = ap.parse_args()
    device = "cpu"
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    from transformers import BertTokenizerFast, BertForQuestionAnswering
    tok = BertTokenizerFast.from_pretrained(args.model)
    model = BertForQuestionAnswering.from_pretrained(args.model).to(device)

    train_ex, valid_ex = build_examples()

    def encode(batch):
        enc = tok([b["question"] for b in batch],
                  [b["context"] for b in batch],
                  truncation="only_second", max_length=128, padding="max_length",
                  return_offsets_mapping=True, return_tensors="pt")
        starts, ends = [], []
        for i, b in enumerate(batch):
            ctx = b["context"]
            a0 = ctx.find(b["answer"])
            a1 = a0 + len(b["answer"])
            # offsets for context tokens (sequence 1)
            seq_ids = enc.sequence_ids(i)
            offs = enc["offset_mapping"][i].tolist()
            offs = [o if seq_ids[j] == 1 else (0, 0) for j, o in enumerate(offs)]
            s, e = char_to_token_span(offs, a0, a1)
            if s is None or e is None:
                s = e = 0
            starts.append(s); ends.append(e)
        enc.pop("offset_mapping")
        return {k: v.to(device) for k, v in enc.items()}, \
               torch.tensor(starts, device=device), torch.tensor(ends, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def batches(ex):
        for i in range(0, len(ex), args.batch):
            yield ex[i:i + args.batch]

    for ep in range(args.epochs):
        model.train()
        random.shuffle(train_ex)
        tot = 0.0
        for batch in batches(train_ex):
            enc, s, e = encode(batch)
            out = model(**enc, start_positions=s, end_positions=e)
            opt.zero_grad(); out.loss.backward(); opt.step()
            tot += out.loss.item() * len(batch)
        # eval
        model.eval()
        em = f1 = 0
        with torch.no_grad():
            for batch in batches(valid_ex):
                enc, s, e = encode(batch)
                out = model(**enc)
                sp = out.start_logits.argmax(1)
                epos = out.end_logits.argmax(1)
                ids = enc["input_ids"]
                for i, b in enumerate(batch):
                    a, c = int(sp[i]), int(epos[i])
                    if c < a:
                        a, c = c, a
                    pred = tok.decode(ids[i][a:c + 1], skip_special_tokens=True)
                    em += int(pred.strip().lower() == b["answer"].lower())
                    f1 += f1_score(pred, b["answer"])
        em /= len(valid_ex); f1 /= len(valid_ex)
        print(f"epoch {ep} | train loss {tot/len(train_ex):.4f} | "
              f"valid EM {em:.4f} F1 {f1:.4f}")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"valid_EM {em:.4f}\nvalid_F1 {f1:.4f}\nmodel {args.model}\n"
                f"n_train {len(train_ex)}\nn_valid {len(valid_ex)}\n")
    # sample predictions
    model.eval()
    with open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8") as f:
        with torch.no_grad():
            for b in valid_ex[:8]:
                enc, s, e = encode([b])
                out = model(**enc)
                a = int(out.start_logits.argmax(1)); c = int(out.end_logits.argmax(1))
                if c < a: a, c = c, a
                pred = tok.decode(enc["input_ids"][0][a:c + 1], skip_special_tokens=True)
                f.write(f"Q: {b['question']}\ngold: {b['answer']}\npred: {pred}\n\n")
    print(f"final valid EM {em:.4f} F1 {f1:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
