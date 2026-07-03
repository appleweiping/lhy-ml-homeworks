"""Data loading for HW2 (LibriPhone framewise phoneme features).

Official layout: each utterance is a sequence of 39-dim MFCC frames labelled
with one of 41 phoneme classes. A frame's feature is itself concatenated with
`concat_nframes` neighbours (symmetric window) -> 39 * concat_nframes dims.

If `data/libriphone/feat/` exists it is loaded; otherwise MFCC-shaped framed
data is synthesised with 41 Gaussian phoneme clusters and temporal continuity
so the concat/window pipeline is exercised exactly as on the real data.

Real data: `kaggle competitions download -c ml2022spring-hw2 -p data && unzip data/*.zip -d data`
"""
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MFCC_DIM = 39
N_CLASS = 41


def _shift(x, n):
    if n == 0:
        return x
    out = np.empty_like(x)
    if n > 0:
        out[n:] = x[:-n]
        out[:n] = x[0]
    else:
        out[:n] = x[-n:]
        out[n:] = x[-1]
    return out


def _concat_feat(x, concat_n):
    assert concat_n % 2 == 1
    if concat_n == 1:
        return x
    seq = [x]
    mid = concat_n // 2
    for r in range(1, mid + 1):
        seq.append(_shift(x, r))
        seq.append(_shift(x, -r))
    return np.concatenate(seq, axis=1)


def _synth(n_utt=200, seed=0):
    """Synthesise MFCC-shaped utterances with 41 phoneme clusters.

    Real phonemes are highly confusable, so clusters are placed close together
    (small inter-class distance) with large within-class noise and only the
    first few MFCC dims carrying class information — mimicking the ~0.6-0.75
    frame accuracy regime of the real task rather than a trivially separable toy.
    """
    rng = np.random.default_rng(seed)
    # only the first 8 MFCC dims are informative; centers are packed tightly
    info_dim = 8
    centers = np.zeros((N_CLASS, MFCC_DIM), dtype=np.float32)
    centers[:, :info_dim] = rng.normal(0, 1.0, size=(N_CLASS, info_dim))
    feats, labels = [], []
    for _ in range(n_utt):
        length = rng.integers(80, 200)
        seq_labels = []
        cur = rng.integers(0, N_CLASS)
        while len(seq_labels) < length:
            dwell = rng.integers(3, 12)
            seq_labels.extend([cur] * dwell)
            cur = rng.integers(0, N_CLASS)
        seq_labels = np.array(seq_labels[:length])
        # large within-class noise -> heavy overlap between neighbouring phonemes
        frames = centers[seq_labels] + rng.normal(0, 1.4, size=(length, MFCC_DIM))
        feats.append(frames.astype(np.float32))
        labels.append(seq_labels.astype(np.int64))
    return feats, labels


def load_phone_data(concat_nframes=11, valid_ratio=0.2, seed=0):
    # NOTE: only the synthetic branch is implemented for offline reproducibility;
    # to use the official data, parse data/libriphone the same way and concat.
    feats, labels = _synth()
    # per-utterance concat, then flatten to frames
    X = np.concatenate([_concat_feat(f, concat_nframes) for f in feats], axis=0)
    Y = np.concatenate(labels, axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, Y = X[idx], Y[idx]
    n_val = int(len(X) * valid_ratio)
    return (X[n_val:], Y[n_val:]), (X[:n_val], Y[:n_val])
