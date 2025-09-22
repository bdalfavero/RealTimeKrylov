"""
Here we are timinng how long it takes to do contractions with a water molecule.
The code here is copied from hf_krylov.py.
"""

from time import perf_counter_ns
import numpy as np
from scipy.linalg import eigh # generalized eigenvalue
from matplotlib import pyplot as plt
import pickle
from joblib import Parallel, delayed # parallelize generation of data
import pyscf
from openfermion.chem import geometry_from_pubchem
# TenPy Imports, TenPy Version 1.0.2
import tenpy as tp
from tenpy.models import CouplingMPOModel, NearestNeighborModel
from tenpy.models import lattice
from tenpy.models.molecular import MolecularModel
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
	for step in range(num_steps):
		print(f"Time = {dt*step}")
		engine.run()
		states.append(psi_gnd.copy())

	return states


def main():
    # TODO Try N_2 as well.
    chidmrg = 2
    chitdvp = 2
    # Get V_ijkl and h_ij for the HF molecule.
    loading_start_time = perf_counter_ns()
    geometry = geometry_from_pubchem("water")
    # geometry = 'H 0 0 0; F 0 0 1.1'
    mol = pyscf.M(
        # atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        atom = geometry,  # in Angstrom
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
    loading_end_time = perf_counter_ns()
    loading_time_elapsed = loading_end_time - loading_start_time
    print(f"Created model in {loading_time_elapsed:1.4e} ns.")

    print(f"Performing DMRG with chi={chidmrg}.")
    dmrg_start_time = perf_counter_ns()
    E, psi = get_gnd_hubbard(mol_model, chidmrg)
    dmrg_end_time = perf_counter_ns()
    dmrg_time_elapsed = dmrg_end_time - dmrg_start_time
    print(f"Got energy E={E}.")
    print(f"Total elapsed time for DMRG: {dmrg_time_elapsed:1.4e} ns.")

    dt = 1e-4
    for k in range(7):
        T = k * dt
        print(f"Evolving ground state with TDVP. T={T}, dt={dt}, chi={chitdvp}.")
        evolve_start_time = perf_counter_ns()
        states = evolve_gnd_hubbard(psi, mol_model, chitdvp)
        evolve_end_time = perf_counter_ns()
        evolve_time_elapsed = evolve_end_time - evolve_start_time
        print(f"Evolved in {evolve_time_elapsed:1.4e} ns.")

if __name__ == "__main__":
	main()
