import pyscf
from openfermion.chem import geometry_from_pubchem

def main():
    geom = geometry_from_pubchem("hydrogen")
    mol = pyscf.M(
        atom = geom,  # in Angstrom
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