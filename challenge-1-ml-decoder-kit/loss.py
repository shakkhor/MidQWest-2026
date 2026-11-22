"""Qubit loss with heralding, added to the kit's memory circuits.

Model (erasure conversion, as on neutral-atom hardware): at the start of every syndrome cycle each
data qubit is lost with probability p_loss. Loss is detected (herald bit = 1) and the atom is
replaced by a fresh one in the maximally mixed state, which is Stim's HERALDED_ERASE: the qubit
gets I, X, Y or Z with probability 1/4 each and the herald fires.

Every herald becomes a detector with coordinate (2,) (the kit uses (0,) for X checks, (1,) for Z
checks), so the detector error model carries the loss information:
  - loss-aware decoding uses the check detectors of the memory basis plus all herald detectors;
  - loss-unaware decoding drops the heralds, so erasures look like extra Pauli noise.

  python loss.py --p 0.004 --p-loss 0.002      # prints the problem sizes
"""
import argparse

import numpy as np
import stim

from circuit import (GROSS_CIRCUITS, _incidence_matrix, _merge_duplicates, _row_numbers, load_circuit)

DATA_QUBITS = list(range(72, 216))      # kit layout: 0-71 X checks, 72-215 data, 216-287 Z checks
X_CHECKS = list(range(0, 72))
HERALD_COORD = 2.0


def add_loss(circuit, p_loss, data_qubits=DATA_QUBITS, before_measuring=X_CHECKS):
    """Flatten the circuit and insert heralded erasure on the data qubits once per cycle.

    The erasure goes right before each measurement of the X checks (the end of every cycle, so
    the loss shows up in the next cycle's checks). Measurement-record targets of detectors and
    observables are re-indexed for the herald records that are inserted.
    """
    flat = circuit.flattened()
    out = stim.Circuit()
    old_to_new = []          # absolute measurement index map
    n_new = 0
    trigger = set(before_measuring)
    for inst in flat:
        name = inst.name
        targets = inst.targets_copy()
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            remapped = []
            for t in targets:
                if t.is_measurement_record_target:
                    old_abs = len(old_to_new) + t.value
                    remapped.append(stim.target_rec(old_to_new[old_abs] - n_new))
                else:
                    remapped.append(t)
            out.append(name, remapped, inst.gate_args_copy())
            continue
        gd = stim.gate_data(name)
        if gd.produces_measurements:
            qubits = {t.value for t in targets if t.is_qubit_target}
            if p_loss > 0 and qubits & trigger:
                out.append("HERALDED_ERASE", data_qubits, [p_loss])
                n_new += len(data_qubits)
                for k in range(len(data_qubits)):
                    out.append("DETECTOR", [stim.target_rec(-len(data_qubits) + k)], [HERALD_COORD])
            n_meas = sum(1 for t in targets if t.is_qubit_target)   # one record per measured qubit
            out.append(inst)
            for _ in range(n_meas):
                old_to_new.append(n_new)
                n_new += 1
        else:
            out.append(inst)
    return out


def detector_groups(circuit):
    """{"X": ids, "Z": ids, "herald": ids} by the first detector coordinate."""
    coords = circuit.get_detector_coordinates()
    groups = {"X": [], "Z": [], "herald": []}
    for d in range(circuit.num_detectors):
        c = coords[d][0]
        groups["X" if c == 0 else "Z" if c == 1 else "herald"].append(d)
    return {k: np.array(v, np.int64) for k, v in groups.items()}


def decoding_ids(circuit, basis, aware):
    """Detector ids a basis decoder sees: its checks, plus the heralds when loss-aware."""
    g = detector_groups(circuit)
    return np.sort(np.concatenate([g[basis], g["herald"]])) if aware else g[basis]


def dem_matrices_ids(circuit, det_ids):
    """Like circuit.dem_matrices, for an explicit list of kept detectors (erasure allowed)."""
    dem = circuit.detector_error_model(approximate_disjoint_errors=True).flattened()
    errors = _merge_duplicates(dem, _row_numbers(det_ids))
    detectors, observables = zip(*errors, strict=True)
    H = _incidence_matrix(detectors, len(det_ids))
    L = _incidence_matrix(observables, circuit.num_observables)
    return H, L.tocsr(), np.array(list(errors.values()))


def loss_circuit(basis, p, p_loss):
    return add_loss(load_circuit(GROSS_CIRCUITS[basis], p), p_loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--p-loss", type=float, default=0.002)
    args = parser.parse_args()
    for basis in "XZ":
        c = loss_circuit(basis, args.p, args.p_loss)
        g = detector_groups(c)
        print(f"{basis} memory: {c.num_detectors} detectors ({len(g['X'])} X, {len(g['Z'])} Z, {len(g['herald'])} heralds)")
        for aware in (False, True):
            H, L, pr = dem_matrices_ids(c, decoding_ids(c, basis, aware))
            print(f"  loss-{'aware' if aware else 'unaware'}: H {H.shape}, {H.nnz} nonzeros, priors {pr.min():.1e}..{pr.max():.1e}")


if __name__ == "__main__":
    main()
