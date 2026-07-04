"""HW5 - Transformer (Neural Machine Translation).

Course: 李宏毅 ML 2022 Spring, HW5.

Implements a full Transformer encoder-decoder (the official architecture:
token+sinusoidal positional embeddings, multi-head self/cross attention, causal
masking, teacher-forced training, greedy decoding) for sequence-to-sequence
machine translation.

The official task translates English<->Chinese from fairseq-preprocessed TED
talks (multi-GB, needs the fairseq toolchain). To run a *real* translation task
on CPU, we train on the real **Multi30k German->English** parallel corpus
(29k sentence pairs, auto-downloaded via HuggingFace `datasets`) — a standard MT
benchmark. We build word-level vocabularies from the training split, train the
Transformer, and report corpus BLEU-4 on the held-out validation split with
greedy decoding.

Reported metrics: validation token accuracy and corpus BLEU-4 on real Multi30k.
"""
import argparse
import collections
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.set_num_threads(3)

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize(s):
    # simple, deterministic word tokenizer (lowercase, split on non-alnum)
    out, cur = [], []
    for ch in s.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur)); cur = []
            if ch.strip():
                out.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def build_vocab(sentences, max_size, min_freq=2):
    cnt = collections.Counter()
    for s in sentences:
        cnt.update(tokenize(s))
    itos = list(SPECIALS)
    for tok, c in cnt.most_common():
        if c < min_freq or len(itos) >= max_size:
            break
        itos.append(tok)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


def encode(s, stoi, max_len):
    ids = [stoi.get(t, UNK) for t in tokenize(s)][: max_len - 2]
    return [BOS] + ids + [EOS]


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, src_stoi, tgt_stoi, max_len):
        self.data = []
        for de, en in pairs:
            s = encode(de, src_stoi, max_len)
            t = encode(en, tgt_stoi, max_len)
            if len(s) > 2 and len(t) > 2:
                self.data.append((s, t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    src = [torch.tensor(s) for s, _ in batch]
    tgt = [torch.tensor(t) for _, t in batch]
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD)
    return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=192, nhead=4, layers=2, ff=384):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=ff, dropout=0.1,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

    def _pad_mask(self, x):
        return x == PAD

    def encode(self, src):
        s = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(s, src_key_padding_mask=self._pad_mask(src))

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        s = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        t = self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        h = self.transformer(
            s, t, tgt_mask=tgt_mask,
            src_key_padding_mask=self._pad_mask(src),
            tgt_key_padding_mask=self._pad_mask(tgt),
            memory_key_padding_mask=self._pad_mask(src),
        )
        return self.out(h)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=40):
        self.eval()
        mem = self.encode(src)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(src.device)
            t = self.pos(self.tgt_emb(ys) * math.sqrt(self.d_model))
            h = self.transformer.decoder(t, mem, tgt_mask=tgt_mask,
                                         memory_key_padding_mask=self._pad_mask(src))
            nxt = self.out(h[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            done = done | (nxt.squeeze(1) == EOS)
            if done.all():
                break
        return ys


def corpus_bleu(refs, hyps):
    """Standard corpus BLEU-4 with brevity penalty over token sequences."""
    weights = [0.25] * 4
    clip_tot = [0] * 4
    cand_tot = [0] * 4
    ref_len = hyp_len = 0

    def ngrams(seq, n):
        return collections.Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))

    for ref, hyp in zip(refs, hyps):
        ref_len += len(ref); hyp_len += len(hyp)
        for n in range(1, 5):
            hc = ngrams(hyp, n); rc = ngrams(ref, n)
            clip_tot[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            cand_tot[n - 1] += max(sum(hc.values()), 1)
    p_log = 0.0
    for n in range(4):
        p = clip_tot[n] / cand_tot[n] if cand_tot[n] else 0
        p_log += weights[n] * math.log(p + 1e-9)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(p_log)


def strip(seq):
    out = []
    for t in seq:
        t = int(t)
        if t == EOS:
            break
        if t not in (PAD, BOS):
            out.append(t)
    return out


def load_multi30k(n_train):
    from datasets import load_dataset
    tr = load_dataset("bentrevett/multi30k", split=f"train[:{n_train}]")
    va = load_dataset("bentrevett/multi30k", split="validation")
    train_pairs = [(r["de"], r["en"]) for r in tr]
    valid_pairs = [(r["de"], r["en"]) for r in va]
    return train_pairs, valid_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--vocab", type=int, default=8000)
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cpu"
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    print("loading real Multi30k (de->en)...")
    train_pairs, valid_pairs = load_multi30k(args.n_train)
    src_stoi, src_itos = build_vocab([p[0] for p in train_pairs], args.vocab)
    tgt_stoi, tgt_itos = build_vocab([p[1] for p in train_pairs], args.vocab)
    print(f"train pairs {len(train_pairs)} valid {len(valid_pairs)} | "
          f"de-vocab {len(src_itos)} en-vocab {len(tgt_itos)}")

    tr_ds = TranslationDataset(train_pairs, src_stoi, tgt_stoi, args.max_len)
    va_ds = TranslationDataset(valid_pairs, src_stoi, tgt_stoi, args.max_len)
    tl = DataLoader(tr_ds, args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(va_ds, args.batch, shuffle=False, collate_fn=collate)

    model = Seq2SeqTransformer(len(src_itos), len(tgt_itos)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    for ep in range(args.epochs):
        model.train()
        tot = n = 0.0
        for s, t in tl:
            s, t = s.to(device), t.to(device)
            logits = model(s, t[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(s); n += len(s)
        # validation: token acc (teacher forced, full) every epoch; greedy-decode
        # BLEU on a subset each epoch (cheap), full-set BLEU only at the end.
        model.eval()
        tok_correct = tok_total = 0
        refs, hyps = [], []
        n_bleu_batches = 3 if ep < args.epochs - 1 else 10 ** 9
        with torch.no_grad():
            for bi, (s, t) in enumerate(vl):
                s, t = s.to(device), t.to(device)
                logits = model(s, t[:, :-1])
                pred_tf = logits.argmax(-1)
                gold = t[:, 1:]
                mask = gold != PAD
                tok_correct += ((pred_tf == gold) & mask).sum().item()
                tok_total += mask.sum().item()
                if bi < n_bleu_batches:
                    dec = model.greedy_decode(s, args.max_len)
                    for i in range(s.size(0)):
                        hyps.append(strip(dec[i])); refs.append(strip(t[i]))
        acc = tok_correct / max(tok_total, 1)
        bleu = corpus_bleu(refs, hyps)
        tag = "BLEU(full)" if ep == args.epochs - 1 else "BLEU(subset)"
        print(f"epoch {ep:2d} | train loss {tot/n:.4f} | "
              f"valid tok-acc {acc:.4f} {tag} {bleu:.4f}")

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"dataset Multi30k de->en\nvalid_token_acc {acc:.4f}\n"
                f"valid_BLEU4 {bleu:.4f}\nsrc_vocab {len(src_itos)}\n"
                f"tgt_vocab {len(tgt_itos)}\nn_train {len(tr_ds)}\n")

    def detok(ids, itos):
        return " ".join(itos[i] for i in ids)

    with open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8") as f:
        for i in range(min(8, len(va_ds))):
            s, t = va_ds[i]
            src_t = torch.tensor(s).unsqueeze(0)
            hyp = strip(model.greedy_decode(src_t, args.max_len)[0])
            f.write(f"DE:  {detok(strip(s), src_itos)}\n")
            f.write(f"REF: {detok(strip(t), tgt_itos)}\n")
            f.write(f"HYP: {detok(hyp, tgt_itos)}\n\n")
    print(f"final valid tok-acc {acc:.4f} BLEU {bleu:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
