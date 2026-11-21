"""Tests of the benchmark harness.

  python -m pytest bench/test_bench.py
"""
import numpy as np
import pandas as pd
import pytest

from baseline import memory_experiments
from bench import make_shots
from bench.run_decoder import parse_config
from bench.stats import per_cycle, summarize, wilson
from circuit import GROSS_CIRCUITS
from decoders import per_cycle_rate


def test_wilson_known_values():
    lo, hi = wilson(10, 100)
    assert lo == pytest.approx(0.05523, abs=1e-4) and hi == pytest.approx(0.17437, abs=1e-4)
    assert wilson(0, 100)[0] == 0.0 and wilson(0, 100)[1] > 0
    assert wilson(100, 100)[1] == pytest.approx(1.0)


def test_per_cycle_matches_kit():
    failed = np.zeros(1000, bool)
    failed[:37] = True
    assert per_cycle(failed.mean()) == pytest.approx(per_cycle_rate(failed, 12)[0])


def test_cached_shots_match_baseline(tmp_path):
    experiments = memory_experiments([GROSS_CIRCUITS["X"], GROSS_CIRCUITS["Z"]], 0.004, 50, 1)
    for basis, experiment in zip("XZ", experiments):
        make_shots.save_shots(0.004, basis, 50, 1, shot_dir=tmp_path)
        det, obs = make_shots.load_shots(0.004, basis, 1, shot_dir=tmp_path)
        assert np.array_equal(det, experiment.det) and np.array_equal(obs, experiment.obs)


def test_parse_config():
    assert parse_config("stop_nconv=1,num_sets=50") == {"stop_nconv": 1, "num_sets": 50}
    assert parse_config("gamma0=0.1") == {"gamma0": 0.1}
    assert parse_config("") == {}


def test_summarize_pairs_x_and_z():
    log = pd.DataFrame({"shot": [0, 1, 0, 1], "basis": list("XXZZ"), "p": 0.004, "decoder": "d", "config": "c",
                        "time_s": [0.1, 0.2, 0.3, 0.4], "fail": [True, False, True, False]})
    s = summarize(log)
    assert (s["fail_X"], s["fail_Z"], s["fail_pairs"], s["pairs"]) == (1, 1, 1, 2)
    assert s["mean_ms"] == pytest.approx(500.0)
