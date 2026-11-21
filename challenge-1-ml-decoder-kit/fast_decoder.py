"""A tiered relay-BP decoder with a learned router, a drop-in replacement for relay-BP.

Stage 1  the first relay leg only (80 iterations of memory BP). Most shots converge here in
         a few milliseconds, and a converged answer is almost never wrong.
Stage 2  relay-BP that stops at its first solution (stop_nconv=1) instead of collecting 5.
Stage 3  only when the router distrusts the stage-2 solution: 4 more relay solutions. The
         lowest-weight solution of all candidates wins, as in full relay-BP (stop_nconv=5).

Without a router the decoder stops after stage 2 (fastest). With a router it spends the
full relay budget only on shots the router flags, so it approaches relay-BP accuracy at a
fraction of its time.
"""
import numpy as np

from decoders import RELAY_SETTINGS


def _relay(H, priors, **changes):
    import relay_bp  # optional: needs a Rust build
    return relay_bp.RelayDecoderF32(H.tocsr(), error_priors=priors, **(RELAY_SETTINGS | changes))


def features(syndrome, result, llr):
    """Router input for one stage-2 result: what makes a first relay solution suspicious."""
    error = np.asarray(result.decoding, bool)
    weight = llr[error].sum()             # -log likelihood of the solution, up to a constant
    synd = float(np.count_nonzero(syndrome))
    return np.array([synd, np.log1p(result.iterations), weight, weight / max(synd, 1.0), error.sum()])


class Router:
    """Logistic regression on features(): probability that the stage-2 solution should not be trusted."""

    def __init__(self, mean, scale, w, b, threshold):
        self.mean, self.scale, self.w, self.b, self.threshold = mean, scale, w, float(b), float(threshold)

    @classmethod
    def fit(cls, X, y, recall=0.95, steps=3000, lr=0.1, l2=1e-3):
        """Fit on features X and labels y (1 = distrust). The threshold keeps `recall` of the positives."""
        mean, scale = X.mean(0), X.std(0) + 1e-9
        Z = (X - mean) / scale
        pos = max(y.mean(), 1e-6)
        sample_w = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))   # balance the rare positive class
        w, b = np.zeros(Z.shape[1]), 0.0
        for _ in range(steps):
            g = sample_w * (_sigmoid(Z @ w + b) - y) / len(y)
            w -= lr * (Z.T @ g + l2 * w)
            b -= lr * g.sum()
        scores = _sigmoid(Z @ w + b)
        threshold = np.quantile(scores[y == 1], 1 - recall) if y.any() else 0.5
        return cls(mean, scale, w, b, threshold)

    def distrust(self, x):
        return _sigmoid(((x - self.mean) / self.scale) @ self.w + self.b) >= self.threshold

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, w=self.w, b=self.b, threshold=self.threshold)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["scale"], d["w"], d["b"], d["threshold"])


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class TieredDecoder:
    """decode(syndrome) -> error vector, like the baselines. `stages` counts where shots ended."""

    def __init__(self, H, priors, router=None):
        self.llr = np.log((1 - priors) / priors)
        self.router = router
        self.first_leg = _relay(H, priors, num_sets=0, stop_nconv=1)
        self.first_solution = _relay(H, priors, stop_nconv=1)
        self.more_solutions = _relay(H, priors, stop_nconv=4) if router else None
        self.stages = np.zeros(3, int)

    def decode(self, syndrome):
        result = self.first_leg.decode_detailed(syndrome)
        self.last_iterations = result.iterations
        if result.success:
            self.stages[0] += 1
            self.last_stage = 1
            return result.decoding
        result = self.first_solution.decode_detailed(syndrome)
        self.last_iterations = result.iterations
        if self.router is None or not self.router.distrust(features(syndrome, result, self.llr)):
            self.stages[1] += 1
            self.last_stage = 2
            return result.decoding
        self.stages[2] += 1
        self.last_stage = 3
        other = np.asarray(self.more_solutions.decode(syndrome), bool)
        mine = np.asarray(result.decoding, bool)
        # Keep the more likely of the two candidates (lower total LLR weight).
        return other if self.llr[other].sum() < self.llr[mine].sum() else mine