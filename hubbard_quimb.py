from typing import Dict, List
import h5py
import numpy as np
import openfermion as of
import qiskit
from quimb_tebd import (
    pauli_sum_to_mpo, get_drmg_ground_state, trotter_circuit_from_psum,
    subspace_matrices, energy_vs_d
)

def main():
    l = 4
    t = 1.0
    u = 6.0
    n_elec = 3
    max_mpo_bond = 100
    tau = 0.2
    steps = 3
    d = 10
    eps = 1e-8

    hamiltonian = of.hamiltonians.fermi_hubbard(l, l, t, u)
    hamiltonian_qubop = of.transforms.jordan_wigner(hamiltonian)
    hamiltonian_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian_qubop)
    qs = hamiltonian_cirq.qubits
    hamiltonian_mpo = pauli_sum_to_mpo(hamiltonian_cirq, qs, max_mpo_bond)

    dmrg_bond_dims = [10, 20, 30]
    dmrg_energies = np.zeros((len(dmrg_bond_dims),), dtype=float)
    dmrg_ground_states: Dict[int, float] = {}
    for i, chi_dmrg in enumerate(dmrg_bond_dims):
        ground_state, energy, occupation = get_drmg_ground_state(
            hamiltonian_mpo, n_elec, chi_dmrg, alpha=10.
        )
        print(f"For chi={chi_dmrg}, energy={energy} and N={occupation}")
        dmrg_ground_states[chi_dmrg] = ground_state
        dmrg_energies[i] = energy

    ev_circuit = trotter_circuit_from_psum(hamiltonian_cirq, tau, steps)
    ev_ckt_transpiled = qiskit.transpile(ev_circuit, basis_gates=["u3", "cx"])
    tebd_bond_dims = [10, 20, 30]
    tebd_energies = np.zeros((len(tebd_bond_dims), d-1), dtype=float)
    for i, chi_tebd in enumerate(tebd_bond_dims):
        h, s = subspace_matrices(
            hamiltonian_mpo, dmrg_ground_states[min(dmrg_bond_dims)], ev_ckt_transpiled,
            chi_tebd, d
        )
        energies = energy_vs_d(h, s, eps)
        print(f"chi={chi_tebd} got energies\n", energies)
        tebd_energies[i, :] = np.array(energies)
    
    f = h5py.File("../data/hubbard_tebd.hdf5")
    f.create_dataset("l", data=l)
    f.create_dataset("t", data=t)
    f.create_dataset("u", data=u)
    f.create_dataset("n_elec", data=n_elec)
    f.create_dataset("max_mpo_bond", data=max_mpo_bond)
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    f.create_dataset("d", data=d)
    f.create_dataset("eps", data=eps)
    f.create_dataset("dmrg_bond_dims", data=dmrg_bond_dims)
    f.create_dataset("dmrg_energies", data=dmrg_energies)
    f.create_dataset("tebd_bond_dims", data=tebd_bond_dims)
    f.create_dataset("tebd_energies", data=tebd_energies)
    f.close()


if __name__ == "__main__":
    main()