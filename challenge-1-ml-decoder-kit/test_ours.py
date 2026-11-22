"""Tests of the TCBP-JOBS decoder and the loss model.  python -m pytest test_ours.py"""
import numpy as np
import pytest

from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
from gf2 import osd0, pack_columns, pack_rows
from jobs import JOBS
from loss import add_loss, detector_groups
from trajbp import bp_fast, bp_trajectories, tanner


@pytest.fixture(scope="module")
def problem():
    circuit = load_circuit(GROSS_CIRCUITS["X"], 0.004)
    H, L, priors = dem_matrices(circuit, "X")
    det, obs = sample(circuit, 24, 123)
    return H, L, priors, det[:, detector_ids(circuit, "X")], obs


def test_numba_bp_matches_numpy(problem):
    H, _, priors, det, _ = problem
    T = tanner(H)
    a, b = bp_trajectories(T, det, priors), bp_fast(T, det, priors)
    assert np.abs(a[0] - b[0]).max() < 1e-4
    assert (a[1] == b[1]).all() and (a[2] == b[2]).all()


def test_bp_decision_satisfies_syndrome_when_converged(problem):
    H, _, priors, det, _ = problem
    _, conv, dec = bp_fast(tanner(H), det, priors)
    assert (((H @ dec[conv].T).T % 2) == det[conv]).all()


def test_pack_rows_matches_columns(problem):
    H, *_ = problem
    cols = pack_columns(H)
    e = np.zeros(H.shape[1], np.uint8)
    e[[3, 500, 4000]] = 1
    s = (H @ e) % 2
    assert (pack_rows(s[None])[0] == cols[3] ^ cols[500] ^ cols[4000]).all()


def test_osd0_and_jobs_corrections_are_valid(problem):
    H, L, priors, det, _ = problem
    T, cols, S = tanner(H), pack_columns(H), pack_rows(det)
    traj, conv, _ = bp_fast(T, det, priors)
    jobs = JOBS(H, L, priors, draws=4, beam=8, depth=10)
    for b in np.flatnonzero(~conv)[:6]:
        score = traj[b, -1].astype(np.float64)
        idx, kstar = osd0(cols, np.argsort(score, kind="stable"), S[b], H.shape[0])
        e0 = np.zeros(H.shape[1], np.uint8)
        e0[idx] = 1
        assert kstar > 0 and (((H @ e0) % 2) == det[b]).all()
        e, cost, gap, _ = jobs.search(score, det[b])
        assert (((H @ e) % 2) == det[b]).all()
        assert cost <= jobs.cost(e0) + 1e-9          # draw 0 starts from OSD-0, the search only improves
        assert abs(cost - jobs.cost(e)) < 1e-6
        assert gap >= 0


def test_loss_insertion_preserves_the_circuit():
    circuit = load_circuit(GROSS_CIRCUITS["X"], 0.004)
    assert circuit.flattened().detector_error_model().approx_equals(
        add_loss(circuit, 0.0).detector_error_model(), atol=1e-12)


def test_checks_fire_only_next_to_heralds():
    lossy = add_loss(load_circuit(GROSS_CIRCUITS["Z"], 0.004).without_noise(), 0.001)
    det = lossy.compile_detector_sampler(seed=5).sample(4000)
    g = detector_groups(lossy)
    quiet = ~det[:, g["herald"]].any(axis=1)
    assert quiet.sum() > 100
    assert not det[quiet][:, np.r_[g["X"], g["Z"]]].any()
