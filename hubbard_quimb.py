from typing import Dict, List
import h5py
from math import sqrt
import numpy as np
from scipy.sparse.linalg import eigsh
import openfermion as of
import qiskit
from quimb_tebd import (
    pauli_sum_to_mpo, get_drmg_ground_state, trotter_circuit_from_psum,
    energy_vs_d, total_number_qubit_operator, get_evolved_states,
    fill_subspace_matrices_mps, mps_to_vector, exact_evolved_states,
    fill_subspace_matrices_vectors
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

    # DMRG energies
    dmrg_bond_dims = [5, 10, 20, 30]
    dmrg_energies = np.zeros((len(dmrg_bond_dims),), dtype=float)
    dmrg_ground_states: Dict[int, float] = {}
    for i, chi_dmrg in enumerate(dmrg_bond_dims):
        ground_state, energy, occupation = get_drmg_ground_state(
            hamiltonian_mpo, n_elec, chi_dmrg, alpha=alpha
        )
        print(f"For chi={chi_dmrg}, energy={energy} and N={occupation}")
        ground_state.normalize()
        dmrg_ground_states[chi_dmrg] = ground_state
        dmrg_energies[i] = energy

    # TEBD Krylov energies
    ev_circuit = trotter_circuit_from_psum(hamiltonian_cirq, tau, steps)
    ev_ckt_transpiled = qiskit.transpile(ev_circuit, basis_gates=["u3", "cx"])
    tebd_bond_dims = dmrg_bond_dims
    ref_state = dmrg_ground_states[5]
    eta = [1e-16, 1e-12, 1e-12, 1e-12]
    tebd_energies = np.zeros((len(tebd_bond_dims), d-1), dtype=float)
    all_tebd_h = np.zeros((len(tebd_bond_dims), d, d), dtype=complex)
    all_tebd_s = np.zeros((len(tebd_bond_dims), d, d), dtype=complex)
    bond_sizes = np.zeros((len(tebd_bond_dims), d, len(hamiltonian_mpo.tensors) - 1), dtype=int)
    for i, chi_tebd in enumerate(tebd_bond_dims):
        states = get_evolved_states(ev_ckt_transpiled, ref_state, d, chi_tebd, None)
        for j, state in enumerate(states):
            bond_sizes[i, j, :] = state.bond_sizes()
        h, s = fill_subspace_matrices_mps(states, hamiltonian_mpo)
        all_tebd_h[i, :, :] = h.copy()
        all_tebd_s[i, :, :] = s.copy()
        energies, _ = energy_vs_d(h, s, method="threshold", eps=eta[i])
        print(f"chi={chi_tebd} got energies\n", energies)
        tebd_energies[i, :] = np.array(energies)
    
    # Krylov with unitary evolution (unitary from circuit)
    ref_vector = mps_to_vector(ref_state)
    states = exact_evolved_states(ev_ckt_transpiled, ref_vector, d)
    h_u, s_u = fill_subspace_matrices_vectors(states, ham_aug_sparse)
    energies_u, _ = energy_vs_d(h_u, s_u, method="threshold", eps=eps)
    
    f = h5py.File("data/hubbard_tebd.hdf5", "w")
    f.create_dataset("l", data=l)
    f.create_dataset("t", data=t)
    f.create_dataset("u", data=u)
    f.create_dataset("n_elec", data=n_elec)
    f.create_dataset("max_mpo_bond", data=max_mpo_bond)
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    f.create_dataset("d", data=d)
    f.create_dataset("eps", data=eps)
    f.create_dataset("exact_energy", data=energy_exact)
    f.create_dataset("dmrg_bond_dims", data=dmrg_bond_dims)
    f.create_dataset("dmrg_energies", data=dmrg_energies)
    f.create_dataset("tebd_bond_dims", data=tebd_bond_dims)
    f.create_dataset("tebd_energies", data=tebd_energies)
    f.create_dataset("energies_u", data=energies_u)
    f.create_dataset("all_tebd_h", data=all_tebd_h)
    f.create_dataset("all_tebd_s", data=all_tebd_s)
    f.create_dataset("h_u", data=h_u)
    f.create_dataset("s_u", data=s_u)
    f.create_dataset("bond_sizes", data=bond_sizes)
    f.close()


if __name__ == "__main__":
    main()