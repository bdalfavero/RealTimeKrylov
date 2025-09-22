import argparse
import json
import numpy as np
from scipy.linalg import eigh # generalized eigenvalue
from matplotlib import pyplot as plt
import pickle
import pyscf
from openfermion.chem import geometry_from_pubchem
# TenPy Imports, TenPy Version 1.0.2
import tenpy as tp
from tenpy.models import CouplingMPOModel, NearestNeighborModel
from tenpy.models import lattice
from tenpy.models.molecular import MolecularModel
# Uncomment the line below if you want TenPy's intrinsic logging 
tp.tools.misc.setup_logging(to_stdout="INFO")

def get_gnd_mol(model, chi, psi_init=None):
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

# Evolve a (ground) state "psi_gnd" using TDVP at fixed bond dimenion chi
# Evolve from time 0 to time T in steps of dt
# Returns tuple of energy and ground state
def evolve_gnd(psi_gnd, model, chi, T=3, dt=0.2):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON input file with parameters.")
    parser.add_argument("output_file", type=str, help="Pickle file with states and parameters.")
    args = parser.parse_args()
    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    molec_name = input_dict["molec_name"]
    chi_dmrg = int(input_dict["chi_dmrg"])
    chi_tdvp = int(input_dict["chi_tdvp"])
    T = float(input_dict["T"])
    dt = float(input_dict["dt"])

    # Get V_ijkl and h_ij for the HF molecule.
    geometry = geometry_from_pubchem(molec_name)
    mol = pyscf.M(
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
    print(mol_model.lat.N_sites_per_ring)

    dmrg_energy, ground_state = get_gnd_mol(mol_model, chi_dmrg)
    states = evolve_gnd(ground_state, mol_model, chi_tdvp, T=T, dt=dt)
    print(f"Final DMRG energy: {dmrg_energy}")
    print(f"Got {len(states)} states.")

    output_dict = {
        "input": input_dict,
        "dmrg_energy": dmrg_energy,
        "ground_state": ground_state,
        "tdvp_states": states,
        "model_mpo": mol_model.H_MPO
    }
    with open(args.output_file, "wb") as f:
        pickle.dump(output_dict, f)


if __name__ == "__main__":
    main()