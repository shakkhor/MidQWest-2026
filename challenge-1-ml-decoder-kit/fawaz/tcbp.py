"""Our version of NVIDIA's trajectory-calibrated BP (TCBP): a small neural network that reads
each fault's BP history and outputs the probability that the fault really happened.

One network is shared by all 8784 faults: it looks at one fault at a time, so it has to learn
general rules ("a belief that dips early and then drifts up is often a true fault"), not facts
about specific faults. Credit: Howe, Maurya et al. (NVIDIA), "TCBP-JOBS".
"""
from __future__ import annotations

import numpy as np

# BP iterations whose belief goes in directly (the first pass runs 80 iterations).
SNAPSHOTS = (0, 1, 2, 4, 8, 16, 32, 48, 64, 79)
FEATURE_NAMES = [f"belief@{t}" for t in SNAPSHOTS] + [
    "lowest belief", "when lowest (0-1)", "highest belief", "average belief",
    "share of rounds below 0", "direction changes (0-1)", "last change", "prior belief",
]


def squash(x):
    """sign(x) * log(1 + |x|): beliefs grow to +-300 with memory BP, this keeps them in +-6."""
    return np.sign(x) * np.log1p(np.abs(x))


def features(history, prior_llr):
    """(n_faults, 18) features from a BP history of shape (iterations, n_faults)."""
    T = history.shape[0]
    steps = np.diff(history, axis=0)
    direction_changes = (np.diff(np.sign(steps), axis=0) != 0).sum(0) / max(T - 2, 1)
    cols = [squash(history[t]) for t in SNAPSHOTS] + [
        squash(history.min(0)),
        history.argmin(0) / (T - 1),
        squash(history.max(0)),
        squash(history.mean(0)),
        (history < 0).mean(0),
        direction_changes,
        squash(steps[-1]),
        prior_llr,
    ]
    return np.column_stack(cols).astype(np.float32)


class TCBPModel:
    """Scaler + MLP (18 -> 16 -> 16 -> 1). Trained on all true faults plus a random share of the
    other faults; `negative_rate` undoes that sampling so the outputs are real probabilities."""

    def __init__(self, hidden=(16, 16), seed=0):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipe = make_pipeline(StandardScaler(), MLPClassifier(hidden, early_stopping=True, max_iter=300,
                                                                   random_state=seed))
        self.negative_rate = 1.0

    def fit(self, X, y, negative_rate):
        self.pipe.fit(X, y)
        self.negative_rate = negative_rate
        return self

    def logit(self, X):
        """log-odds that each fault happened, corrected for the negative subsampling."""
        p = np.clip(self.pipe.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p)) + np.log(self.negative_rate)

    def probability(self, X):
        return 1 / (1 + np.exp(-self.logit(X)))

    def n_weights(self):
        mlp = self.pipe[-1]
        return sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
