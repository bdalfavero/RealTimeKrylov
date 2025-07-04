"""Real-time Krylov for chemical Hamiltonians."""

from typing import List, Tuple
import argparse
import argparse
import json
import pickle
import numpy as np
from scipy.linalg import eigh
import pyscf
import tenpy as tp
from tenpy.networks.mps import MPS
from tenpy.models.molecular import MolecularModel

def partially_filled_state(n_sites: int, n_electrons: int) -> List[str]:
    """Make a state where the first n_electrons/2 orbitals are full, and the rest are empty.
    If n_electrons is odd, then the next unoccupied site will be filled with a spin-down
    electron.
    E.g. For n_sites = 4, n_electrons = 3, returns ['full', 'down', 'empty', 'empty']."""

    assert n_electrons < 2 * n_sites
    n_full = n_electrons // 2
    if n_electrons % 2 != 0:
        # Odd number of electrons - Extra spin down in the next site.
        n_half_full = 1
    else:
        n_half_full = 0
    n_empty = n_sites - n_full - n_half_full
    product_state = ["full"] * n_full + ["down"] * n_half_full + ["empty"] * n_empty
    assert len(product_state) == n_sites
    return product_state


def get_ground_state(model: MolecularModel, n_electrons: int, max_bond: int) -> Tuple[MPS, float]:
    """Get the DMRG ground state of the molecular model."""

    # TODO Different initial state?
    # product_state = ["up", "down"] * (len(model.lat.mps_sites()) // 2) # start in semi-Néel state 
    n_sites = len(model.lat.mps_sites())
    product_state = partially_filled_state(n_sites, n_electrons)
    psi = tp.MPS.from_product_state(model.lat.mps_sites(), product_state)
    dmrg_params = {'mixer': True, 'trunc_params': {'chi_max': max_bond, 'svd_min': 1e-9},
        'max_E_err': 1e-9, 'max_S_err': 1e-6, 'min_sweeps': 20, 'max_sweeps': 50, 'max_trunc_err': None,
        'max_N_sites_per_ring': None}
    engine = tp.TwoSiteDMRGEngine(psi, model, dmrg_params)
    print(engine.options)
    e_ground, psi_ground = engine.run()
    return psi_ground, e_ground


def evolve_state(psi: MPS, model: MolecularModel, chi: float, T: float = 3, dt: float = 0.2) -> List[MPS]:
    """Evolve the state for total time T with steps of size dt."""

    # parameters for each step of TDVP
    num_steps = int(T / dt)
    time_params = {'start_time': 0, 'dt': dt, 'N_steps': 1,
        'trunc_params': {'chi_max': chi, 'svd_min': 1.e-10, 'trunc_cut': None},
        'max_N_sites_per_ring': None}

    # evolve the inputted state psi in place
    engine = tp.TwoSiteTDVPEngine(psi, model, time_params)

    # Save a copy of the evolved state at each time step
    states = [psi.copy()] # initial state at t = 0
    for step in range(num_steps):
        print(f"Time = {dt*step}")
        engine.run()
        states.append(psi.copy())

    return states


def subspace_matrices(psi_dmrg: MPS, model: MolecularModel, chi: int, T: float, dt: float):
    """Compute subspace matrices."""
    
    # Evolve the initial state with maximum bond dimension chi_time
    states = evolve_state(psi_dmrg.copy(), model, chi=chi, T=T, dt=dt)
    
    # create overlap and target matrices
    N = len(states)
    S, H = [np.zeros((N, N), dtype=complex) for _ in range(2)]

    # fill off-diagonal elements with overlaps and hamiltonian expectation values
    for i in range(N):
        for j in range(i+1, N):
            S[i, j] = states[i].overlap(states[j]) # < vi | vj >
            H[i, j] = tp.MPOEnvironment(states[i], model.H_MPO, states[j]).full_contraction(0) # < vi | H | vj >
    H += H.conj().T
    S += S.conj().T

    # fill diagonal elements 
    for i in range(N):
        S[i, i] = states[i].overlap(states[i]).real
        H[i, i] = model.H_MPO.expectation_value(states[i]).real

    return H, S


def krylov_energy_offset(H: np.ndarray, S: np.ndarray) -> float:
    """Use the 'add a small value' method to get the ground state energy
    from the Krylov subspace matrices H and S."""

    N = S.shape[0] # size of krylov subspace
    energies = []
    
    # Try to keep successively more krylov states and plot energies that result
    for keep in range(1, N + 1):
        # keep only parts of target/overlap matrices
        print(f"Keeping {keep} Krylov states")
        H0, S0 = H[:keep, :keep], S[:keep, :keep] 
        
        # CHECK THAT OVERLAPS MATRIX IS NOT BADLY CONDITIONED
        # If it is good enough, solve generalized eigenvalue problem in krylov subspace
        # H | v > = E S | v >
        svals = np.linalg.svd(S0, compute_uv=False)
        cond = svals.max() / svals.min()
        print("Condition Number:", cond)
        try:
            vals, vecs = eigh(H0, S0)
        except np.linalg.LinAlgError:
            print("HAD CONVERGENCE PROBLEM")
            eps = 1e-12
            correction = eps*np.eye(S0.shape[0])
            vals, vecs = eigh(H0, S0 + correction)
        print(f"Current energy {np.min(vals)}")

        # record difference btwn krylov ansatz and the true ground state energy 
        energies.append(np.min(vals))
    return np.min(energies)


def threshold_eigenvalues(h: np.ndarray, s: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Remove all eigenvalues below a positive threshold eps.
    See Epperly et al. sec. 1.2."""

    # Build a matrix whose columns correspond to the positive eigenvectors of s.
    evals, evecs = eigh(s)
    positive_evals = []
    positive_evecs = []
    for i, ev in enumerate(evals):
        assert abs(ev.imag) < 1e-7
        if ev.real > eps:
            positive_evals.append(ev.real)
            positive_evecs.append(evecs[:, i])
    pos_evec_mat = np.vstack(positive_evecs).T
    # Project h and s into this subspace.
    new_s =  pos_evec_mat.conj().T @ s @ pos_evec_mat
    new_h = pos_evec_mat.conj().T @ h @ pos_evec_mat
    return new_h, new_s


def krylov_energy_thresholded(H: np.ndarray, S: np.ndarray, eps: float) -> float:
    """Get the ground state energy by projecting the H and S matrices onto the space
    spanned by the eigenvectors of S, the associated eigenvalues of which are above some threshold
    value epsilon."""

    energies = []
    for keep in range(1, S.shape[0]):
        H0 = H[:keep, :keep]
        S0 = S[:keep, :keep]
        H_new, S_new = threshold_eigenvalues(H0, S0, eps)
        eigvals, eigvecs = eigh(H_new, S_new)
        energy = np.min(eigvals)
        print(f"For dimension {keep} got energy {energy}.")
        energies.append(energy)
    return np.min(energies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", type=str, help="JSON file with parameters.")
    parser.add_argument("output_file", type=str, help="JSON file for simulation output.")
    args = parser.parse_args()

    # Read parameters from JSON file.
    with open(args.params_file) as f:
        input_dict = json.load(f)

    # Get V_ijkl and h_ij for the HF molecule.
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
        symmetry = True,
    )
    orb = mol.RHF().run().mo_coeff
    v_ijkl = mol.intor('int2e', aosym='s1')
    h_ij = mol.intor('int1e_nuc') + mol.intor('int1e_kin')
    print("One- and two-body tensor dimensions:")
    print(h_ij.shape)
    print(v_ijkl.shape)
    # Make a TeNPy molecular model
    params = {"one_body_tensor": h_ij, "two_body_tensor": v_ijkl}
    mol_model = MolecularModel(params)
    print(mol_model.lat.N_sites_per_ring)

    # Get ground state energy from DRMG.
    max_bond = input_dict["chi"]
    n_electrons = input_dict["n_electrons"]
    ground_state, dmrg_energy = get_ground_state(mol_model, n_electrons, max_bond)
    print("DMRG energy =", dmrg_energy)
    with open('hf_ground_state.pkl', 'wb') as f:
        pickle.dump(ground_state, f)

    H, S = subspace_matrices(
        ground_state, mol_model,
        input_dict["chi"], input_dict["T"], input_dict["dt"]
    )
    subspace_output = {
        "H": H, "S": S
    }
    with open("subspace_matrices.pkl", "wb") as f:
        pickle.dump(subspace_output, f)

    # Get energy with eigenvalue thresholding.
    krylov_thresholded_energy = krylov_energy_thresholded(H, S, input_dict["eps"])

    output_dict = {
        "input": input_dict,
        "dmrg_energy": dmrg_energy,
        "kyrlov_energy": krylov_thresholded_energy
    }
    with open(args.output_file, 'w') as f:
        json.dump(output_dict, f)


if __name__ == "__main__":
    main()