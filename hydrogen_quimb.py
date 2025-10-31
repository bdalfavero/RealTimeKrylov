import h5py
import numpy as np
import scipy as sp
from scipy.sparse.linalg import norm
import openfermion as of
from openfermionpyscf import run_pyscf
from quimb_tebd import (
    pauli_sum_to_mpo, get_drmg_ground_state, trotter_circuit_from_psum,
    energy_vs_d, get_evolved_states, fill_subspace_matrices_mps,
    exact_evolved_states, mps_to_vector, fill_subspace_matrices_vectors
)

def main():
    molec = "hydrogen"
    basis = "sto-3g"
    n_elec = 2
    geometry = of.chem.geometry_from_pubchem(molec)
    multiplicity = 1
    dmrg_max_bond = 5 
    tebd_max_bond = 40
    steps = 10
    eps = 1e-12
    d = 15

    # Get the Hamiltonian.
    molecule = of.chem.MolecularData(
        geometry, basis, multiplicity
    )
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
    print(f"HF energy:", molecule.hf_energy)
    print(f"FCI energy:", molecule.fci_energy)
    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian_qubop = of.transforms.jordan_wigner(hamiltonian)
    hamiltonian_psum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian_qubop)
    ham_sparse = of.linalg.get_sparse_operator(hamiltonian_qubop)
    qs = hamiltonian_psum.qubits
    hamiltonian_mpo = pauli_sum_to_mpo(hamiltonian_psum, qs, dmrg_max_bond)

    ham_norm = norm(ham_sparse)
    tau = np.pi / ham_norm

    # Run DMRG
    ground_state, energy, number = get_drmg_ground_state(hamiltonian_mpo, n_elec, dmrg_max_bond)
    print(f"DMRG ground state has energy {energy} and occupation {number}.")

    # Do TEBD Krylov with the DMRG reference state.
    ev_circuit = trotter_circuit_from_psum(hamiltonian_psum, tau, steps)
    states = get_evolved_states(ev_circuit, ground_state, d, tebd_max_bond, None)
    h, s = fill_subspace_matrices_mps(states, hamiltonian_mpo)
    energies, num_kept = energy_vs_d(h, s, method="threshold", eps=eps)

    # Do Krylov with the unitary of the circuit and a vector derived from the DMRG ground state.
    gs_vector = mps_to_vector(ground_state)
    print("Norm of vector version of ground state:", sp.linalg.norm(gs_vector))
    u_states = exact_evolved_states(ev_circuit, gs_vector, d)
    h_u, s_u = fill_subspace_matrices_vectors(u_states, ham_sparse)
    # breakpoint()
    energies_u, num_kept_u = energy_vs_d(h_u, s_u, method="threshold", eps=eps)
    
    f = h5py.File("data/h2_results.hdf5", "w")
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.create_dataset("h_u", data=h_u)
    f.create_dataset("s_u", data=s_u)
    f.create_dataset("energies", data=energies)
    f.create_dataset("energies_u", data=energies_u)
    f.create_dataset("num_kept", data=num_kept)
    f.create_dataset("num_kept_u", data=num_kept_u)
    f.create_dataset("hf_energy", data=molecule.hf_energy)
    f.create_dataset("fci_energy", data=molecule.fci_energy)
    f.create_dataset("dmrg_energy", data=energy)
    f.close()


if __name__ == "__main__":
    main()