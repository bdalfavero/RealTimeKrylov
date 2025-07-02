"""DMRG for chemical models with TeNPy. Hamiltonians from HamLib."""

from typing import List, Tuple
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

    # product_state = ["up", "down"] * (len(model.lat.mps_sites()) // 2) # start in semi-Néel state 
    n_sites = len(model.lat.mps_sites())
    product_state = partially_filled_state(n_sites, n_electrons)
    psi = tp.MPS.from_product_state(model.lat.mps_sites(), product_state)
    dmrg_params = {'mixer': True, 'trunc_params': {'chi_max': max_bond, 'svd_min': 1e-9},
        'max_E_err': 1e-9, 'max_S_err': 1e-6, 'min_sweeps': 20, 'max_sweeps': 50, 'max_trunc_err': None}
    engine = tp.TwoSiteDMRGEngine(psi, model, dmrg_params)
    e_ground, psi_ground = engine.run()
    return psi_ground, e_ground


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
    print(mol_model.lat.N_sites_per_ring)
    # Run DMRG
    print("Running DMRG")
    max_bond = 10
    n_electrons = 10 # PubChem says the formal charge of HF is 0.
    ground_state, energy = get_ground_state(mol_model, n_electrons, max_bond)
    print("DMRG energy =", energy)

if __name__ == "__main__":
    main()
