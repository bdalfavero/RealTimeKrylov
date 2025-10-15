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
    ham_augmented_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_augmented)
    ham_augmented_mpo = pauli_sum_to_mpo(ham_augmented_cirq, qs, max_mpo_bond)

    chi_dmrg = 5
    ground_state, energy, occupation = get_drmg_ground_state(
        ham_augmented_mpo, n_elec, chi_dmrg, alpha=alpha
    )

    step_vals = np.logspace(0, 4, num=4)
    h_errs = np.zeros((step_vals.size,), dtype=float)
    s_errs = np.zeros((step_vals.size,), dtype=float)
    all_h_toep = np.zeros((d, d, step_vals.size), dtype=complex)
    all_h_non_toep = np.zeros((d, d, step_vals.size), dtype=complex)
    all_s_toep = np.zeros((d, d, step_vals.size), dtype=complex)
    all_s_non_toep = np.zeros((d, d, step_vals.size), dtype=complex)
    for i, steps in enumerate(step_vals):
        print(f"On {i} out of {len(step_vals)}")
        ev_circuit = trotter_circuit_from_psum(hamiltonian_cirq, tau, int(steps))
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
        h_errs[i] = la.norm(h_full - h_toep)
        s_errs[i] = la.norm(s_full - s_toep)
        all_h_toep[:, :, i] = h_toep
        all_h_non_toep[:, :, i] = h_full
        all_s_toep[:, :, i] = s_toep
        all_s_non_toep[:, :, i] = s_full
    
    f = h5py.File("data/subspace_errors.hdf5", "w")
    f.create_dataset("steps", data=step_vals)
    f.create_dataset("h_errs", data=h_errs)
    f.create_dataset("s_errs", data=s_errs)
    f.create_dataset("all_h_toep", data=all_h_toep)
    f.create_dataset("all_h_non_toep", data=all_h_non_toep)
    f.create_dataset("all_s_toep", data=all_s_toep)
    f.create_dataset("all_s_non_toep", data=all_s_non_toep)
    f.close()

if __name__ == "__main__":
    main()