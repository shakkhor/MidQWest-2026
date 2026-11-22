"""TCBP: trajectory-calibrated BP (after NVIDIA's TCBP-JOBS paper), re-implemented for the kit.

One small MLP, shared by every fault, reads a fault's BP trajectory (T marginal LLRs plus four
summary features) and outputs the probability that the fault fired. OSD then orders columns by
that probability instead of by BP's final LLR.

Train:  python tcbp.py --shots 4000 --out tcbp_model.npz
"""
import argparse
import time

import numpy as np
import torch

from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from trajbp import bp_trajectories, tanner

CLIP = 15.0


def features(traj, priors):
    """(B, T, n) trajectories and (B, n) or (n,) priors -> (B, n, T + 4) float32 features."""
    B, T, n = traj.shape
    x = np.transpose(traj, (0, 2, 1)) / CLIP                              # (B, n, T)
    argmin = traj.argmin(axis=1)[..., None] / T                           # (B, n, 1)
    slopes = np.diff(traj, axis=1)
    sign_changes = (np.diff(np.sign(slopes), axis=1) != 0).sum(axis=1)[..., None] / T
    final_slope = (slopes[:, -1] / CLIP)[..., None]
    pri = np.broadcast_to(np.asarray(priors, np.float32), (B, n))
    prior_llr = (np.log((1 - pri) / pri) / CLIP)[..., None]
    return np.concatenate([x, argmin, sign_changes, final_slope, prior_llr], axis=2).astype(np.float32)


class TCBP:
    """The trained MLP in NumPy: (T + 4) -> 16 -> 16 -> 1, ReLU, logistic output."""

    def __init__(self, path):
        z = np.load(path)
        self.w = [z[f"w{i}"] for i in range(3)]
        self.b = [z[f"b{i}"] for i in range(3)]

    def logits(self, feats):
        h = feats
        for i in range(2):
            h = np.maximum(h @ self.w[i] + self.b[i], 0)
        return (h @ self.w[2] + self.b[2])[..., 0]

    def scores(self, traj, priors):
        """OSD ordering scores, smaller = more likely: -logit = log((1 - p̂) / p̂)."""
        return -self.logits(features(traj, priors))


def training_rows(H, priors, shots, rng, iters=10, neg_keep=0.05, batch=None):
    """Sample errors straight from H with jittered, rescaled DEM priors and record BP features.

    Each shot draws a noise scale (p from 0.002 to 0.006 relative to the circuit's 0.004) and a
    per-fault log-normal jitter, so the model learns trajectory shapes across operating points.
    Only shots where BP does not converge are kept, since those are the shots OSD sees.
    Returns X, y, sample weights.
    """
    T = tanner(H)
    Hc = H.tocsc()
    batch = batch or max(8, min(256, int(2e7 / (H.shape[1] * iters))))
    Xs, ys, ws = [], [], []
    done = 0
    while done < shots:
        b = min(batch, shots - done)
        scale = np.exp(rng.uniform(np.log(0.5), np.log(1.5), (b, 1)))
        rates = np.clip(priors[None] * scale * np.exp(rng.normal(0, 0.3, (b, len(priors)))), 1e-5, 0.1)
        errors = (rng.random(rates.shape) < rates).astype(np.uint8)
        syndromes = (Hc @ errors.T).T % 2
        traj, conv, _ = bp_trajectories(T, syndromes, rates, iters=iters)
        keep = ~conv
        f = features(traj[keep], rates[keep]).reshape(-1, iters + 4)
        y = errors[keep].reshape(-1)
        sel = (y == 1) | (rng.random(len(y)) < neg_keep)
        Xs.append(f[sel]); ys.append(y[sel]); ws.append(np.where(y[sel] == 1, 1.0, 1 / neg_keep))
        done += b
    return np.concatenate(Xs), np.concatenate(ys).astype(np.float32), np.concatenate(ws).astype(np.float32)


def fit(X, y, w, epochs=4, seed=1):
    """Weighted logistic loss, Adam. Returns the layer weights as NumPy arrays."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 16), torch.nn.ReLU(),
                                torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    Xt, yt, wt = map(torch.from_numpy, (X, y, w))
    n = len(Xt)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 8192):
            idx = perm[i:i + 8192]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                model(Xt[idx])[:, 0], yt[idx], reduction="none") * wt[idx]).sum() / wt[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch}: loss {total / n:.5f}")
    layers = [m for m in model if isinstance(m, torch.nn.Linear)]
    return {**{f"w{i}": l.weight.detach().numpy().T.copy() for i, l in enumerate(layers)},
            **{f"b{i}": l.bias.detach().numpy().copy() for i, l in enumerate(layers)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", type=int, default=4000, help="training shots per basis")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="tcbp_model.npz")
    parser.add_argument("--full-dem", action="store_true", help="train on the full DEM (all 1728 detectors)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    parts = []
    start = time.time()
    for basis in "XZ":
        H, _, priors = dem_matrices(load_circuit(GROSS_CIRCUITS[basis], 0.004), None if args.full_dem else basis)
        parts.append(training_rows(H, priors, args.shots, rng, args.iters))
        print(f"basis {basis}: {len(parts[-1][0]):,} rows ({time.time() - start:.0f} s)")
    X, y, w = (np.concatenate(z) for z in zip(*parts))
    np.savez(args.out, **fit(X, y, w, seed=args.seed), iters=args.iters)
    print(f"saved {args.out} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
