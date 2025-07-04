"""Run MP2 and CCSD for a molecule."""

import pyscf

def main():
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
        symmetry = True,
    )
    hf = mol.RHF.run()
    # Run MP2.
    hf.MP2().run()
    # Run CCSD.
    cc = pyscf.cc.CCSD(hf).run()
    print('Total CCSD energy', cc.e_tot)
    et = cc.ccsd_t()
    print('CCSD(T) total energy', cc.e_tot + et)
    mf = mol.UHF().run()
    mycc = hf.CISD().run()
    print('UCISD correlation energy', mycc.e_corr)


if __name__ == "__main__":
    main()