"""Sample shots once and cache them, so every decoder sees exactly the same shots.

  python -m bench.make_shots --p 0.003 0.004 0.005 --shots 10000 --seed 1

The file shots/p{p}_{basis}_seed{seed}.npz holds the shots of the run seed `seed`, sampled
exactly as baseline.py samples them: X memory with sampler seed `seed`, Z memory with
`seed + 1`. Stim's output depends on the number of shots requested, so take prefixes of
one file for smaller runs instead of sampling fewer shots.
"""
import argparse
from pathlib import Path

import numpy as np

from circuit import GROSS_CIRCUITS, detector_ids, load_circuit, sample

SHOT_DIR = Path(__file__).resolve().parent.parent / "shots"
BASES = "XZ"


def shot_path(p, basis, seed, shot_dir=SHOT_DIR):
    return Path(shot_dir) / f"p{p}_{basis}_seed{seed}.npz"


def sample_basis(p, basis, shots, seed):
    """(basis detectors, observables) exactly as baseline.memory_experiments produces them."""
    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    det, obs = sample(circuit, shots, seed + BASES.index(basis))
    return det[:, detector_ids(circuit, basis)], obs


def save_shots(p, basis, shots, seed, shot_dir=SHOT_DIR):
    det, obs = sample_basis(p, basis, shots, seed)
    path = shot_path(p, basis, seed, shot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, det=np.packbits(det, axis=1), n_det=det.shape[1], obs=obs,
                        p=p, basis=basis, seed=seed, sampler_seed=seed + BASES.index(basis))
    return path


def load_shots(p, basis, seed, shots=None, shot_dir=SHOT_DIR):
    """(detectors, observables) of the cached file; the first `shots` rows if given."""
    with np.load(shot_path(p, basis, seed, shot_dir)) as f:
        det = np.unpackbits(f["det"], axis=1, count=int(f["n_det"]))
        obs = f["obs"]
    return det[:shots], obs[:shots]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, nargs="+", default=[0.003, 0.004, 0.005])
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, nargs="+", default=[1], help="1 test, 1000 train, 2000 validation")
    args = parser.parse_args()
    for seed in args.seed:
        for p in args.p:
            for basis in BASES:
                print(save_shots(p, basis, args.shots, seed))


if __name__ == "__main__":
    main()
