"""DMRG for the HF molecule with Block2. One- and two-body integrals done with PySCF."""

import numpy as np
from pyscf import gto, scf
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

def main():
    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
        symmetry = True,
    )
    # orb = mol.RHF().run().mo_coeff
    # v_ijkl = mol.intor('int2e', aosym='s1')
    # h_ij = mol.intor('int1e_nuc') + mol.intor('int1e_kin')
    bond_dims = [250] * 4 + [500] * 4
    noises = [1e-4] * 4 + [1e-5] * 4 + [0]
    thrds = [1e-10] * 8
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
        ncore=0, ncas=None, g2e_symm=8)

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
    ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
    energy = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
        thrds=thrds, iprint=1)
    print('DMRG energy = %20.15f' % energy)

    pdm1 = driver.get_1pdm(ket)
    pdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)
    print('Energy from pdms = %20.15f' % (np.einsum('ij,ij->', pdm1, h1e)
        + 0.5 * np.einsum('ijkl,ijkl->', pdm2, driver.unpack_g2e(g2e)) + ecore))

    impo = driver.get_identity_mpo()
    expt = driver.expectation(ket, mpo, ket) / driver.expectation(ket, impo, ket)
    print('Energy from expectation = %20.15f' % expt)

if __name__ == "__main__":
    main()