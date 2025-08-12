"""Get plot of energy vs. bond dim with curves for different
Kyrlov subspace dimensions."""

import numpy as np
from scipy.linalg import eigh # generalized eigenvalue
from matplotlib import pyplot as plt
import pandas as pd
import pickle
from pathos.pools import ProcessPool
import pyscf
# TenPy Imports, TenPy Version 1.0.2
import tenpy as tp
from tenpy.models import CouplingMPOModel, NearestNeighborModel
from tenpy.models import lattice
from tenpy.models.molecular import MolecularModel
# Uncomment the line below if you want TenPy's intrinsic logging 
tp.tools.misc.setup_logging(to_stdout="INFO")

def get_gnd_hf(model, chi, psi_init=None):
    # By default, start with a state that has half-filling (one electron per site)
    if psi_init is None:
        product_state = ["up", "down"] * (len(model.lat.mps_sites()) // 2) # start in semi-Néel state 
        n_sites = len(model.lat.mps_sites())
        if len(product_state) != n_sites:
            product_state.append("up")
        # product_state = ["up", "down"]  * (Lx * Ly // 2) # start in semi-Néel state 
        psi = tp.MPS.from_product_state(model.lat.mps_sites(), product_state)
    else:
        psi = psi_init

    # Set up DMRG parameters and precision (bond dimension beyond chi is truncated)
    dmrg_params = {'mixer': True, 'trunc_params': {'chi_max': chi, 'svd_min': 1e-9},
        'max_E_err': 1e-9, 'max_S_err': 1e-6, 'min_sweeps': 20, 'max_sweeps': 50, 'max_trunc_err': None,
        'max_N_sites_per_ring': None}
    engine = tp.TwoSiteDMRGEngine(psi, model, dmrg_params)
    
    # Run DMRG and return energy and ground state
    E, psi = engine.run()
    print(f"E = {E}")
    
    return (E, psi)


def evolve_gnd_hf(psi_gnd, model, chi, T=3, dt=0.2):
    # parameters for each step of TDVP
    num_steps = int(T / dt)
    time_params = {'start_time': 0, 'dt': dt, 'N_steps': 1,
        'trunc_params': {'chi_max': chi, 'svd_min': 1.e-10, 'trunc_cut': None}, 'max_N_sites_per_ring': None}
    
    # evolve the inputted state psi_gnd in place
    engine = tp.TwoSiteTDVPEngine(psi_gnd, model, time_params)

    # Save a copy of the evolved state at each time step
    states = [psi_gnd.copy()] # initial state at t = 0
    for step in range(num_steps):
        print(f"Time = {dt*step}")
        engine.run()
        states.append(psi_gnd.copy())

    return states


def subspace_matrices(psi_dmrg, model_ref, chi_time, T=3, dt=0.2):
    states = evolve_gnd_hf(psi_dmrg.copy(), model_ref, chi=chi_time, T=T, dt=dt)
    
    # create overlap and target matrices
    N = len(states)
    S, H = [np.zeros((N, N), dtype=complex) for _ in range(2)]

    # fill off-diagonal elements with overlaps and hamiltonian expectation values
    for i in range(N):
        for j in range(i+1, N):
            S[i, j] = states[i].overlap(states[j]) # < vi | vj >
            H[i, j] = tp.MPOEnvironment(states[i], model_ref.H_MPO, states[j]).full_contraction(0) # < vi | H | vj >
    H += H.conj().T
    S += S.conj().T

    # fill diagonal elements 
    for i in range(N):
        S[i, i] = states[i].overlap(states[i]).real
        H[i, i] = model_ref.H_MPO.expectation_value(states[i]).real
    return H, S


def energy_vs_d(H, S):
    """Get energy at different subsapce dimensions given the subspace matrices."""

    N = H.shape[0]
    # Try to keep successively more krylov states and plot energies that result
    energies = []
    ds = range(1, N + 1)
    for keep in ds:
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

        # record difference btwn krylov ansatz and the true ground state energy 
        energies.append(vals[0]) 
    return ds, energies


def main():
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

    # Do DMRG at a range of bond dimensions.
    chi_dmrg = list(range(20, 50, 5))
    pool = ProcessPool(nodes=8)
    results = pool.map(lambda chi: get_gnd_hf(mol_model, chi, psi_init=None), chi_dmrg)
    dmrg_energies = {chi: res[0] for chi, res in zip(chi_dmrg, results)}
    dmrg_states = {chi: res[1] for chi, res in zip(chi_dmrg, results)}

    # Make the DMRG data into a dataframe with chi, d=0, and energy.
    records = []
    for chi, energy in dmrg_energies.items():
        records.append((chi, 0, energy))
    df_dmrg = pd.DataFrame.from_records(records, columns=["chi", "d", "energy"])

    # Get energies with TDVP-Krylov.
    d_max = 5
    chi_time = max(chi_dmrg)
    T = 3
    dt = 0.2

    def energy_callback(chi, psi_dmrg):
        """Returns a datafame with of energy vs. d, with
        a column for chi."""

        H, S = subspace_matrices(psi_dmrg, mol_model, chi_time, T, dt)
        ds, energies = energy_vs_d(H, S)
        records = [(chi, d, e) for d, e in zip(ds, energies)]
        df = pd.DataFrame.from_records(records, columns=["chi", "d", "energy"])
        return df
    
    pool = ProcessPool(nodes=8)
    dfs = pool.map(lambda chi: energy_callback(chi, dmrg_states[chi]), chi_dmrg)

    # Combine the data into one big dataframe.
    big_df = pd.concat([df_dmrg] + dfs)
    big_df.index.name = "i"
    big_df.to_csv("hf_energy_vs_bond_dim.csv")

if __name__ == "__main__":
    main()
