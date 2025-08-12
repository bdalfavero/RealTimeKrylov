"""Get the DMRG ground state at bond dim chi, then do TDVP for time t at chi' > chi.
Get the von Neumann entropy averaged over the bipartitions."""

from typing import List, Tuple
import numpy as np
from scipy.linalg import eigh # generalized eigenvalue
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import pyscf
from pathos.pools import ProcessPool
# TenPy Imports, TenPy Version 1.0.2
import tenpy as tp
from tenpy.models import CouplingMPOModel, NearestNeighborModel
from tenpy.models import lattice
from tenpy.models.molecular import MolecularModel
from tenpy.networks.mps import MPS
# Uncomment the line below if you want TenPy's intrinsic logging 
tp.tools.misc.setup_logging(to_stdout="INFO")

# get ground state of Hubbard model using DMRG at dixed bond dimension chi
# Lx, Ly are system size, chi = DMRG bond dimension, U = Hubbard repulsion, psi_init is initial state
# Returns tuple of energy and ground state
def get_gnd_hubbard(model, chi, psi_init=None):
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

# Evolve a (ground) state "psi_gnd" of Hubbard model using TDVP at fixed bond dimenion chi
# Evolve from time 0 to time T in steps of dt
# Returns tuple of energy and ground state
def evolve_gnd_hubbard(psi_gnd, model, chi, T=3, dt=0.2):
    # parameters for each step of TDVP
    num_steps = int(T / dt)
    time_params = {'start_time': 0, 'dt': dt, 'N_steps': 1,
        'trunc_params': {'chi_max': chi, 'svd_min': 1.e-10, 'trunc_cut': None}, 'max_N_sites_per_ring': None}
    
    # evolve the inputted state psi_gnd in place
    engine = tp.TwoSiteTDVPEngine(psi_gnd, model, time_params)

    # Save a copy of the evolved state at each time step
    states = [psi_gnd.copy()] # initial state at t = 0
    times = [0.]
    for step in range(num_steps):
        print(f"Time = {dt*step}")
        times.append(dt * step)
        engine.run()
        states.append(psi_gnd.copy())

    return times, states


def average_entropy(mps: MPS) -> float:
    """Get the von Neumann entropy of an MPS averaged over all biparitions.
    In an MPS the sites are laid out in a line s1, s2, ..., s_L, so we only consider biparitions
    of the form {s1, ..., s_k}, {s_{k+1}, ..., s_L}."""

    entropies = mps.entanglement_entropy()
    return np.average(entropies)


def entropy_vs_time(times: List[float], states: List[MPS]) -> Tuple[List[float], List[float]]:
    """von Neumann entropy vs. time for the given time evolution."""

    entropies: List[float] = []
    for t, mps in zip(times, states):
        entropies.append(average_entropy(mps))
    return times, entropies


def main():
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

    # Get the dmrg ground state for each chi.
    print("Starting DMRG.")
    chis = [10, 20, 30]
    pool = ProcessPool(nodes=4)
    results = pool.map(lambda chi: get_gnd_hubbard(mol_model, chi), chis)
    dmrg_ground_states = {chi: result[1] for chi, result in zip(chis, results)}

    print("Computing entropy.")
    # Compute entropy vs. time for each bond dimension
    entropy_results = {}
    for chi, mps in dmrg_ground_states.items():
        times, mpses = evolve_gnd_hubbard(mps, mol_model, np.max(chis))
        times, entropies = entropy_vs_time(times, mpses)
        entropy_results[chi] = (times, entropies)
    
    # Convert to a dataframe for output.
    records = []
    for chi, (times, entropies) in entropy_results.items():
        for t, s in zip(times, entropies):
            records.append((chi, t, s))
    df = pd.DataFrame.from_records(records, columns=["chi", "t", "entropy"])
    df.index.name = "i"
    df.to_csv("hf_entropies.csv")


if __name__ == "__main__":
    main()