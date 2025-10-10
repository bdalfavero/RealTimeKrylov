from typing import Dict, List
import h5py
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
import openfermion as of
import qiskit
from quimb_tebd import (
    pauli_sum_to_mpo, get_drmg_ground_state, trotter_circuit_from_psum,
    subspace_matrices, energy_vs_d, total_number_qubit_operator
)

def main():
    l = 2
    t = 1.0
    u = 6.0
    n_elec = 3
    max_mpo_bond = 100
    tau = 0.2
    steps = 3
    d = 5
    eps = 1e-8
    alpha = 10.

    hamiltonian = of.hamiltonians.fermi_hubbard(l, l, t, u)
    hamiltonian_qubop = of.transforms.jordan_wigner(hamiltonian)
    hamiltonian_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian_qubop)
    qs = hamiltonian_cirq.qubits
    hamiltonian_mpo = pauli_sum_to_mpo(hamiltonian_cirq, qs, max_mpo_bond)
    total_number = total_number_qubit_operator(len(qs))
    augment_term = alpha * (total_number - n_elec) ** 2
    ham_augmented = hamiltonian_qubop + augment_term

    chi_dmrg = 5
    ground_state, energy, occupation = get_drmg_ground_state(
        hamiltonian_mpo, n_elec, chi_dmrg, alpha=alpha
    )

    ev_circuit = trotter_circuit_from_psum(hamiltonian_cirq, tau, steps)
    ev_ckt_transpiled = qiskit.transpile(ev_circuit, basis_gates=["u3", "cx"])
    chi_tebd=5
    h_full, s_full = subspace_matrices(
        hamiltonian_mpo, ground_state, ev_ckt_transpiled,
        chi_tebd, d, method="full"
    )
    h_toep, s_toep = subspace_matrices(
        hamiltonian_mpo, ground_state, ev_ckt_transpiled,
        chi_tebd, d, method="Toeplitz"
    )
    print(la.norm(h_full - h_toep))
    print(la.norm(s_full - s_toep))
    breakpoint()

if __name__ == "__main__":
    main()