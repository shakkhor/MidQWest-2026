"""Checks for our trajectory-recording BP.  python -m pytest test_bp_trajectory.py -v"""
import numpy as np
import pytest
import relay_bp

from bp_trajectory import TrajectoryBP
import _kit  # noqa: F401,E402  (makes the kit's circuit/baseline/decoders importable)
from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
from decoders import RELAY_SETTINGS


@pytest.fixture(scope="module")
def problem():
    circuit = load_circuit(GROSS_CIRCUITS["X"], 0.004)
    H, L, priors = dem_matrices(circuit, "X")
    det, _ = sample(circuit, 40, seed=7)
    return H, priors, det[:, detector_ids(circuit, "X")]


@pytest.mark.parametrize("gamma", [None, RELAY_SETTINGS["gamma0"]])
def test_every_iteration_matches_library(problem, gamma):
    """Our beliefs after t iterations equal the library's BP run for t iterations."""
    H, priors, det = problem
    ours = TrajectoryBP(H, priors, alpha=1.0, gamma=gamma)
    extra = {} if gamma is None else {"gamma0": gamma}
    for syndrome in det[:3]:
        history, _ = ours.run(syndrome, 10)
        for t in (1, 4, 10):
            lib = relay_bp.MinSumBPDecoderF64(H.tocsr(), error_priors=priors, max_iter=t, alpha=1.0, **extra)
            np.testing.assert_allclose(history[t - 1], lib.decode_detailed(syndrome).posterior_ratios, atol=1e-8)


def test_converges_exactly_when_relay_first_pass_does(problem):
    """Our 80-iteration memory BP solves the same shots as relay-BP's first pass."""
    H, priors, det = problem
    ours = TrajectoryBP(H, priors, alpha=1.0, gamma=RELAY_SETTINGS["gamma0"])
    first_pass = relay_bp.RelayDecoderF64(H.tocsr(), error_priors=priors, **(RELAY_SETTINGS | {"num_sets": 0}))
    for syndrome in det:
        _, converged_at = ours.run(syndrome, RELAY_SETTINGS["pre_iter"], stop_when_solved=True)
        assert (converged_at >= 0) == first_pass.decode_detailed(syndrome).success
