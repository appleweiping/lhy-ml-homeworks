"""Data loading for HW1.

If the official Kaggle CSV (`covid.train.csv` / `covid.test.csv`) is present in
`data/`, it is used directly. Otherwise a dataset with the *exact* official
column layout is synthesised: 40 US-state one-hot columns + 3 daily blocks of
survey features (behaviour indicators, belief indicators, mental-health, and the
previous day's tested_positive), with a realistic non-linear target. This lets
the network train on real-shaped data on any machine.

Download the real data with:
    kaggle competitions download -c ml2022spring-hw1 -p data && unzip -o data/*.zip -d data
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
N_STATES = 40
DAYS = 3
# per-day survey feature groups (matches the official 18-per-day layout)
PER_DAY = 18


def _official_paths():
    tr = os.path.join(DATA_DIR, "covid.train.csv")
    te = os.path.join(DATA_DIR, "covid.test.csv")
    return tr, te


def _load_official():
    tr, te = _official_paths()
    train = pd.read_csv(tr).values[:, 1:].astype(np.float32)  # drop id
    test = pd.read_csv(te).values[:, 1:].astype(np.float32)
    return train, test


def _synth(n_train=2700, n_test=900, seed=0):
    """Synthesise data matching the official feature layout.

    Feature vector = [40 state one-hot] + [DAYS x PER_DAY survey features].
    The last survey column of each day is that day's tested_positive rate; the
    target is the 4th day's rate, a smooth non-linear function of the window.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    n_feat = N_STATES + DAYS * PER_DAY

    # state one-hot
    states = rng.integers(0, N_STATES, size=n)
    onehot = np.zeros((n, N_STATES), dtype=np.float32)
    onehot[np.arange(n), states] = 1.0

    # per-state baseline positivity + a slow random walk over the 3 days
    state_base = rng.uniform(5, 25, size=N_STATES)[states]
    survey = np.zeros((n, DAYS * PER_DAY), dtype=np.float32)
    pos = np.zeros((n, DAYS), dtype=np.float32)
    prev = state_base.copy()
    for d in range(DAYS):
        block = rng.normal(0.5, 0.15, size=(n, PER_DAY - 1)).astype(np.float32)
        # correlate a few behaviour features with positivity
        prev = np.clip(prev + rng.normal(0, 1.5, size=n) + 4 * (block[:, 0] - 0.5), 1, 60)
        pos[:, d] = prev
        survey[:, d * PER_DAY : d * PER_DAY + (PER_DAY - 1)] = block
        survey[:, d * PER_DAY + (PER_DAY - 1)] = prev

    x = np.concatenate([onehot, survey], axis=1).astype(np.float32)

    # target: momentum of last-day + small trend + state effect + noise
    trend = pos[:, 2] - pos[:, 0]
    target = (
        0.85 * pos[:, 2]
        + 0.10 * trend
        + 2.0 * np.sin(survey[:, -PER_DAY] * 3.0)  # non-linear survey coupling
        + rng.normal(0, 1.0, size=n)
    )
    target = np.clip(target, 0, None).astype(np.float32)

    data = np.concatenate([x, target[:, None]], axis=1).astype(np.float32)
    assert data.shape[1] == n_feat + 1
    train = data[:n_train]
    test = data[n_train:, :-1]  # test has no label column
    return train, test


def load_covid_data():
    tr, te = _official_paths()
    if os.path.exists(tr) and os.path.exists(te):
        print("[data] using official Kaggle CSVs")
        return _load_official()
    os.makedirs(DATA_DIR, exist_ok=True)
    print("[data] official CSVs not found -> synthesising official-layout dataset")
    return _synth()
