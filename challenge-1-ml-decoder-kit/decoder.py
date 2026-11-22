"""TCBP-JOBS decoder for one memory experiment: BP -> TCBP scores -> JOBS on non-converged shots."""
import numpy as np

from jobs import JOBS
from tcbp import TCBP
from trajbp import bp_fast, tanner


class TcbpJobsDecoder:
    """decode(syndrome) -> correction, compatible with decoders.score; decode_full adds diagnostics."""

    def __init__(self, H, L, priors, model_path="tcbp_model.npz", iters=10, **jobs_settings):
        self.H, self.L, self.priors = H, L, np.asarray(priors, np.float64)
        self.tanner = tanner(H)
        self.model = TCBP(model_path)
        self.iters = iters
        self.jobs = JOBS(H, L, priors, **jobs_settings)
        self.w = self.jobs.w

    def decode_full(self, syndrome):
        """(correction, info) for one syndrome. info: converged, cost, gap, kstar."""
        syndrome = np.asarray(syndrome, np.uint8)
        traj, conv, dec = bp_fast(self.tanner, syndrome[None], self.priors, iters=self.iters)
        if conv[0]:
            e = dec[0]
            return e, {"converged": True, "cost": float(self.w[e.astype(bool)].sum()), "gap": np.inf, "kstar": 0}
        score = self.model.scores(traj, self.priors)[0]
        e, cost, gap, kstar = self.jobs.search(score, syndrome)
        return e, {"converged": False, "cost": cost, "gap": gap, "kstar": kstar,
                   "coset": self.jobs.last_coset[0], "margin": self.jobs.last_coset[1]}

    def decode(self, syndrome):
        return self.decode_full(syndrome)[0]


class TwoStageDecoder:
    """BP -> small JOBS (stage 1); if its confidence gap is below tau, full JOBS (stage 2).

    Stage 2 returns the cheaper of its own and stage 1's corrections. With tau=None both stages
    run on every non-converged shot and decode_full reports both, for tuning tau offline.
    """

    def __init__(self, H, L, priors, model_path="tcbp_model.npz", iters=10, tau=None,
                 stage1=None, stage2=None):
        self.H, self.L, self.priors = H, L, np.asarray(priors, np.float64)
        self.tanner = tanner(H)
        self.model = TCBP(model_path)
        self.iters = iters
        self.tau = tau
        self.fast = JOBS(H, L, priors, **(stage1 or {"draws": 3, "beam": 8, "depth": 15, "patience": 4}))
        self.full = JOBS(H, L, priors, **(stage2 or {}))
        self.w = self.full.w

    def decode_full(self, syndrome):
        import time
        syndrome = np.asarray(syndrome, np.uint8)
        t0 = time.perf_counter()
        traj, conv, dec = bp_fast(self.tanner, syndrome[None], self.priors, iters=self.iters)
        if conv[0]:
            e = dec[0]
            c = float(self.w[e.astype(bool)].sum())
            return e, {"converged": True, "cost": c, "gap": np.inf, "kstar": 0, "e1": e, "gap1": np.inf,
                       "t1": time.perf_counter() - t0, "t2": 0.0}
        score = self.model.scores(traj, self.priors)[0]
        e1, c1, g1, kstar = self.fast.search(score, syndrome)
        t1 = time.perf_counter() - t0
        info = {"converged": False, "kstar": kstar, "e1": e1, "gap1": g1, "t1": t1, "t2": 0.0}
        if self.tau is not None and g1 >= self.tau:
            info.update(cost=c1, gap=g1)
            return e1, info
        t = time.perf_counter()
        e2, c2, g2, _ = self.full.search(score, syndrome)
        e, c = (e2, c2) if c2 <= c1 else (e1, c1)
        info.update(cost=c, gap=g2, t2=time.perf_counter() - t)
        return e, info

    def decode(self, syndrome):
        return self.decode_full(syndrome)[0]
