"""Run NVIDIA's own TCBP-JOBS (CUDA-Q QEC 0.9.0, nv-qldpc-decoder, preset "tcbp-jobs") on our exact shots.

Kaggle notebook, GPU T4 x2, internet on. Upload kaggle/gross_p0.004_shots.npz as a dataset, then:
  cell 1: !pip install -q --upgrade cuda-quantum-cu12 cudaq-qec   (then restart the session)
  cell 2: this file.
Result on 2026-11-21 (10,000 shots per basis, p = 0.004):
  NVIDIA TCBP-JOBS 41 failed pairs (3.42e-4 / cycle) | ours 41 | relay-BP 103
  NVIDIA vs ours: 4 vs 4 discordant pairs (McNemar p = 1): identical accuracy.
  NVIDIA single-shot latency on a T4: 21.0 ms per X+Z pair; model fit ~135 s per basis on the T4.
"""
import time

import numpy as np
import scipy.sparse as sp
import cudaq_qec as qec
from cudaq_qec.plugins import nv_qldpc
from scipy.stats import binomtest

z = np.load("/kaggle/input/datasets/hummi3/gross-shots/gross_p0.004_shots.npz")


def problem(b):
    H = sp.csr_matrix((np.ones(len(z[f"{b}_H_rows"]), np.uint8), (z[f"{b}_H_rows"], z[f"{b}_H_cols"])),
                      shape=tuple(z[f"{b}_H_shape"]))
    L = sp.csr_matrix((np.ones(len(z[f"{b}_L_rows"]), np.uint8), (z[f"{b}_L_rows"], z[f"{b}_L_cols"])),
                      shape=tuple(z[f"{b}_L_shape"]))
    det = np.unpackbits(z[f"{b}_det"], axis=1)[:, : int(z[f"{b}_ndet"])]
    return H, L, z[f"{b}_priors"].astype(np.float64), det, z[f"{b}_obs"]


nv_failed, nv_valid, lat = {}, {}, {}
for b in "XZ":
    H, L, pri, det, obs = problem(b)
    t = time.time()
    model = nv_qldpc.init_tcbp_model(H, seed=1)
    fit_s = time.time() - t
    dec = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=pri, **nv_qldpc.preset("tcbp-jobs", tcbp_model=model))
    dec.decode(det[0].astype(np.float64).tolist())
    ts = []
    for s in det[:300]:
        t0 = time.perf_counter()
        dec.decode(s.astype(np.float64).tolist())
        ts.append(time.perf_counter() - t0)
    lat[b] = np.array(ts) * 1e3
    t = time.time()
    E = []
    for i in range(0, len(det), 1024):
        E += [np.asarray(r.result) for r in dec.decode_batch(det[i:i + 1024].astype(np.float64).tolist())]
    run_s = time.time() - t
    E = (np.array(E) > 0.5).astype(np.uint8)
    nv_valid[b] = (((H @ E.T).T % 2) == det).all(axis=1)
    nv_failed[b] = (((L @ E.T).T % 2) != obs).any(axis=1) | ~nv_valid[b]
    print(f"{b}: model fit {fit_s:.1f}s | {len(det)} shots in {run_s:.1f}s | valid {nv_valid[b].mean():.4f} | "
          f"NVIDIA {nv_failed[b].sum()} | ours {z[f'{b}_ours_failed'].sum()} | relay {z[f'{b}_relay_failed'].sum()} | "
          f"single-shot mean {lat[b].mean():.1f} ms")

nv = nv_failed["X"] | nv_failed["Z"]
ours = z["X_ours_failed"] | z["Z_ours_failed"]
relay = z["X_relay_failed"] | z["Z_relay_failed"]
print(f"failed pairs: NVIDIA {nv.sum()} | ours {ours.sum()} | relay-BP {relay.sum()}")
for name, other in [("ours", ours), ("relay-BP", relay)]:
    a, c = int((nv & ~other).sum()), int((other & ~nv).sum())
    print(f"NVIDIA vs {name}: only NVIDIA {a}, only {name} {c}, McNemar p = {binomtest(a, a + c).pvalue if a + c else 1:.3g}")
np.savez("/kaggle/working/nvidia_tcbpjobs_p0.004.npz", X_failed=nv_failed["X"], Z_failed=nv_failed["Z"],
         X_lat_ms=lat["X"], Z_lat_ms=lat["Z"])
