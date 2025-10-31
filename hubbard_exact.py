from typing import Dict, List
import h5py
from math import sqrt
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
import openfermion as of
import qiskit
from quimb_tebd import (
    pauli_sum_to_mpo, get_drmg_ground_state, trotter_circuit_from_psum,
    subspace_matrices, energy_vs_d, total_number_qubit_operator, fill_subspace_matrices_toeplitz
)

def main():
    l = 2
    t = 1.0
    u = 6.0
    n_elec = 3
    max_mpo_bond = 100
    tau = 0.2
    steps = 3
    d = 15
    eps = 1e-8
    alpha = 10.

    hamiltonian = of.hamiltonians.fermi_hubbard(l, l, t, u)
    hamiltonian_qubop = of.transforms.jordan_wigner(hamiltonian)
    hamiltonian_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian_qubop)
    qs = hamiltonian_cirq.qubits
    hamiltonian_mpo = pauli_sum_to_mpo(hamiltonian_cirq, qs, max_mpo_bond)
    ham_matrix = hamiltonian_cirq.matrix(qs)

    # Get exact energy (if the Hamilonian is small!)
    total_number = total_number_qubit_operator(len(qs))
    augment_term = alpha * (total_number - n_elec) ** 2
    ham_augmented = hamiltonian_qubop + augment_term
    ham_aug_sparse = of.linalg.get_sparse_operator(ham_augmented)
    eigvals, eigvecs = eigsh(ham_aug_sparse, which="SA")
    energy_exact = np.min(eigvals)
    exact_ground_state = eigvecs[:, np.argmin(eigvals.real)]
    print(f"Exact energy = {energy_exact}")

    ptb_state = np.zeros((2 ** len(qs),), dtype=complex)
    idx = (1 << n_elec) - 1
    ptb_state[idx] = 1.0
    r = 0.4
    ref_state = sqrt(1 - r) * exact_ground_state + sqrt(r) * ptb_state

    overlaps = []
    mat_elems = []
    for dd in range(d):
        u = la.expm(-1j * tau * dd * ham_matrix)
        overlap = np.vdot(ref_state, u @ ref_state)
        mat_elem = np.vdot(ref_state, ham_matrix @ u @ ref_state)
        overlaps.append(overlap)
        mat_elems.append(mat_elem)
    h, s = fill_subspace_matrices_toeplitz(mat_elems, overlaps)

    energies = energy_vs_d(h, s, eps=eps, method="threshold")
    print(energies)

if __name__ == "__main__":
    main()