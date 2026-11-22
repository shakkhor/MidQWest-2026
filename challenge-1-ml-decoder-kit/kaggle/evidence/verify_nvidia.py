"""Re-check the NVIDIA comparison from the saved per-shot results (no GPU needed).

  cd challenge-1-ml-decoder-kit && python kaggle/evidence/verify_nvidia.py

Inputs:
  kaggle/evidence/nvidia_tcbpjobs_p0.004.npz   NVIDIA TCBP-JOBS (CUDA-Q QEC 0.9.0, Kaggle T4) per-shot failures
  kaggle/gross_p0.004_shots.npz                 the exact shots, with our and relay-BP's per-shot failures
  results/tcbpjobs_p0.004.npz, results/relay_p0.004.npz   our evaluation records of the same shots
"""
import os

import numpy as np
from scipy.stats import binomtest

here = os.path.dirname(os.path.abspath(__file__))
kit = os.path.dirname(os.path.dirname(here))
nv = np.load(os.path.join(here, "nvidia_tcbpjobs_p0.004.npz"))
shots = np.load(os.path.join(kit, "kaggle", "gross_p0.004_shots.npz"))
ours = np.load(os.path.join(kit, "results", "tcbpjobs_p0.004.npz"))["failed"]
relay = np.load(os.path.join(kit, "results", "relay_p0.004.npz"))["failed"]

# the shots file carries the same per-shot failures as our evaluation records
assert (shots["X_ours_failed"] == ours[:, 0]).all() and (shots["Z_ours_failed"] == ours[:, 1]).all()
assert (shots["X_relay_failed"] == relay[:, 0]).all() and (shots["Z_relay_failed"] == relay[:, 1]).all()

pairs = {"NVIDIA TCBP-JOBS (their code, T4 GPU)": nv["X_failed"] | nv["Z_failed"],
         "ours (TCBP-JOBS re-implementation, CPU)": ours.any(axis=1),
         "relay-BP (kit baseline)": relay.any(axis=1)}
print(f"p = 0.004, {len(ours)} shots per basis (X and Z memory), failed pairs and rate per cycle:")
for name, f in pairs.items():
    print(f"  {name:42s} {int(f.sum()):4d}   {1 - (1 - f.mean()) ** (1 / 12):.2e}")
a = pairs["NVIDIA TCBP-JOBS (their code, T4 GPU)"]
for name in list(pairs)[1:]:
    b = pairs[name]
    x, y = int((a & ~b).sum()), int((b & ~a).sum())
    print(f"NVIDIA vs {name}: only NVIDIA fails {x}, only the other fails {y}, "
          f"McNemar p = {binomtest(x, x + y).pvalue if x + y else 1.0:.3g}")
print(f"NVIDIA single-shot latency (T4): X {nv['X_lat_ms'].mean():.1f} ms + Z {nv['Z_lat_ms'].mean():.1f} ms per pair")
