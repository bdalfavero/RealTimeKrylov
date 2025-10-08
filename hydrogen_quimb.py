import openfermion as of
from openfermionpyscf import run_pyscf
from quimb_tebd import (
    pauli_sum_to_mpo, get_drmg_ground_state, trotter_circuit_from_psum,
    subspace_matrices, energy_vs_d
)

def main():
    molec = "hydrogen"
    basis = "sto-3g"
    n_elec = 2
    geometry = of.chem.geometry_from_pubchem(molec)
    multiplicity = 1
    dmrg_max_bond = 2
    tebd_max_bond = 40
    tau = 1.0
    steps = 10
    eps = 1e-8
    d = 10

    # Get the Hamiltonian.
    molecule = of.chem.MolecularData(
        geometry, basis, multiplicity
    )
    molecule = run_pyscf(molecule, run_scf=1)
    print(f"HF energy:", molecule.hf_energy)
    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian_qubop = of.transforms.jordan_wigner(hamiltonian)
    hamiltonian_psum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian_qubop)
    qs = hamiltonian_psum.qubits
    hamiltonian_mpo = pauli_sum_to_mpo(hamiltonian_psum, qs, dmrg_max_bond)

    # Run DMRG
    ground_state, energy, number = get_drmg_ground_state(hamiltonian_mpo, n_elec, dmrg_max_bond)
    print(f"DMRG ground state has energy {energy} and occupation {number}.")

    # Do Krylov with the DMRG reference state.
    ev_circuit = trotter_circuit_from_psum(hamiltonian_psum, tau, steps)
    h, s = subspace_matrices(hamiltonian_mpo, ground_state, ev_circuit, tebd_max_bond, d)
    energies = energy_vs_d(h, s, eps)
    print("Krylov energies:\n", energies)


if __name__ == "__main__":
    main()