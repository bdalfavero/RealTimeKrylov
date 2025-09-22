import argparse
import pickle
import h5py
import numpy as np
from scipy.linalg import eigh
import tenpy as tp

def subspace_matrices(states, ham_mpo):
    N = len(states)
    S, H = [np.zeros((N, N), dtype=complex) for _ in range(2)]

    # fill off-diagonal elements with overlaps and hamiltonian expectation values
    for i in range(N):
        for j in range(i+1, N):
            S[i, j] = states[i].overlap(states[j]) # < vi | vj >
            H[i, j] = tp.MPOEnvironment(states[i], ham_mpo, states[j]).full_contraction(0) # < vi | H | vj >
    H += H.conj().T
    S += S.conj().T

    # fill diagonal elements 
    for i in range(N):
        S[i, i] = states[i].overlap(states[i]).real
        H[i, i] = ham_mpo.expectation_value(states[i]).real
    return H, S


def krylov_energy(H, S):
    N = S.shape[0] # size of krylov subspace
    energies = [] # Difference between krylov ansatz and true ground state energy
    
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

        # record difference btwn krylov ansatz and the true ground state energy 
        energies.append(vals[0]) 
    return energies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("states_file", type=str, help="Pickle file with DMRG+TDVP states.")
    parser.add_argument("output_file", type=str, help="HDF5 file for output of subspace matrices and energies.")
    args = parser.parse_args()

    with open(args.states_file, "rb") as f:
        states_output = pickle.load(f)
    tdvp_states = states_output["tdvp_states"]
    ham_mpo = states_output["model_mpo"]

    H, S = subspace_matrices(tdvp_states, ham_mpo)
    energies = krylov_energy(H, S)
    print(f"Lowest energy: {np.min(energies)}")

    f = h5py.File(args.output_file, "w")
    f.create_dataset("input_filename", data=args.states_file)
    f.create_dataset("H", data=H)
    f.create_dataset("S", data=S)
    f.create_dataset("energies", data=np.array(energies))
    f.close()

if __name__ == "__main__":
    main()