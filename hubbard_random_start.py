"""Compare Krylov and DMRG energies for the Hubbard model when the reference state
for Krylov is a randomly-chosen MPS. We want to reproduce the 'bond dimension wall'."""

from typing import Tuple
import json
import numpy as np
from scipy.linalg import eigh # generalized eigenvalue
from matplotlib import pyplot as plt
import pickle
from pathos.pools import ProcessPool

# TenPy Imports, TenPy Version 1.0.2
import tenpy as tp
from tenpy.models import CouplingMPOModel, NearestNeighborModel
from tenpy.models import lattice
from tenpy.networks import MPS
# Uncomment the line below if you want TenPy's intrinsic logging 
tp.tools.misc.setup_logging(to_stdout="INFO")

# Exact energy of ground state of:
# 4 x 2 square lattice Hubbard model with U=8, t=1, at half-filling
E_gnd = -3.02592280569069 # lowest energy
E_antignd = 35.0258 # highest energy state

# Create a Square Lattice Hubbard Model as an MPO in Tenpy
class HubbardSquare2D(CouplingMPOModel):
	
	# initialize sites of the model to have spin-1/2 fermions
	# conserve particle number but not spin, work at half-filling
	def init_sites(self, model_params):
		conserve = model_params.get('cons_N', 'N') 
		filling = model_params.get('filling', 1.)
		site = tp.networks.site.SpinHalfFermionSite(cons_N='N', cons_Sz=None, filling=1)
		return site

	# Initialize a square lattice of dimensions Lx x Ly
	# Open boundary conditions by default in both directions 
	def init_lattice(self, model_params):
		Lx = model_params.get('Lx', 4)
		Ly = model_params.get('Ly', 2)
		bc_x = model_params.get('bc_x', 'open') 
		bc_y = model_params.get('bc_y', 'open')
		bc_MPS = model_params.get('bc_MPS', 'finite')
		lattice = tp.models.lattice.Square(Lx, Ly, site=self.init_sites(model_params), 
							bc=[bc_x, bc_y], bc_MPS=bc_MPS)
		# ordering of sites is up-down snake: start bottom left, end top right
		return lattice

	# Define the Hamiltonian
	def init_terms(self, model_params):
		t = model_params.get('t', 1.0) # hopping
		U = model_params.get('U', 8.0) # onsite Hubbard repulsion
		mu = model_params.get('mu', 0) # chemical potential 
		for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
			self.add_coupling(t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)  # h.c. Cdagger_down C_down
			self.add_coupling(t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)  # h.c. Cdagger_up C_up
		for v in range(len(self.lat.unit_cell)):
			self.add_onsite(U, v, 'NuNd') # Hubbard n_up n_down term
		for v in range(len(self.lat.unit_cell)):
			self.add_onsite(mu, v, 'Nu') # chemical potential
			self.add_onsite(mu, v, 'Nd') # chemical potential

# get ground state of Hubbard model using DMRG at dixed bond dimension chi
# Lx, Ly are system size, chi = DMRG bond dimension, U = Hubbard repulsion, psi_init is initial state
# Returns tuple of energy and ground state
def get_gnd_hubbard(Lx, Ly, chi, U=8, psi_init=None):
	params = {'Lx': Lx, 'Ly':Ly, 'U':U, 'cons_N':'N', 'filling':1}
	model = HubbardSquare2D(params)

	# By default, start with a state that has half-filling (one electron per site)
	if psi_init is None:
		product_state = ["up", "down"]  * (Lx * Ly // 2) # start in semi-Néel state 
		psi = tp.MPS.from_product_state(model.lat.mps_sites(), product_state)
	else:
		psi = psi_init

	# Set up DMRG parameters and precision (bond dimension beyond chi is truncated)
	dmrg_params = {'mixer': True, 'trunc_params': {'chi_max': chi, 'svd_min': 1e-9},
		'max_E_err': 1e-9, 'max_S_err': 1e-6, 'min_sweeps': 20, 'max_sweeps': 50, 'max_trunc_err': None}
	engine = tp.TwoSiteDMRGEngine(psi, model, dmrg_params)
	
	# Run DMRG and return energy and ground state
	E, psi = engine.run()
	print(f"E = {E}")
	
	return (E, psi)

# Evolve a (ground) state "psi_gnd" of Hubbard model using TDVP at fixed bond dimenion chi
# Evolve from time 0 to time T in steps of dt
# Returns tuple of energy and ground state
def evolve_gnd_hubbard(psi_gnd, Lx, Ly, chi, U=8, T=3, dt=0.2):
	params = {'Lx': Lx, 'Ly':Ly, 'U':U, 'cons_N':'N', 'filling':1}
	model = HubbardSquare2D(params)
	
	# parameters for each step of TDVP
	num_steps = int(T / dt)
	time_params = {'start_time': 0, 'dt': dt, 'N_steps': 1,
		'trunc_params': {'chi_max': chi, 'svd_min': 1.e-10, 'trunc_cut': None} }
	
	# evolve the inputted state psi_gnd in place
	engine = tp.TwoSiteTDVPEngine(psi_gnd, model, time_params)

	# Save a copy of the evolved state at each time step
	states = [psi_gnd.copy()] # initial state at t = 0
	for step in range(num_steps):
		print(f"Time = {dt*step}")
		engine.run()
		states.append(psi_gnd.copy())

	return states


def make_data(psi, Lx, Ly, chi, U, T, dt):
	print(f"\n\nSTARTING chi={chi}\n\n")
	
	model_ref = HubbardSquare2D({'Lx': Lx, 'Ly': Ly, 'U':U, 'mu':0}) # Hamiltonian to calculate overlaps
	# Evolve the initial state with maximum bond dimension chi_time
	states = evolve_gnd_hubbard(psi.copy(), Lx=Lx, Ly=Ly, chi=chi, U=U, T=T, dt=dt)
	
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


def krylov_energy_thresholded(H: np.ndarray, S: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get the ground state energy by projecting the H and S matrices onto the space
    spanned by the eigenvectors of S, the associated eigenvalues of which are above some threshold
    value epsilon."""

    ds = []
    energies = []
    for keep in range(1, S.shape[0]):
        ds.append(keep)
        H0 = H[:keep, :keep]
        S0 = S[:keep, :keep]
        H_new, S_new = threshold_eigenvalues(H0, S0, eps)
        eigvals, eigvecs = eigh(H_new, S_new)
        energy = np.min(eigvals)
        print(f"For dimension {keep} got energy {energy}.")
        energies.append(energy)
    return ds, energies


def random_krylov(psi, Lx, Ly, chi, U, T, dt, eps):
	"""Starting from a random state psi, get the Krylov energies and save them to a file."""

	# psi_large = psi.enlarge_chi([chi] * (Lx * Ly) + [0])
	H, S = make_data(psi.copy(), Lx, Ly, chi, U, T, dt)
	ds, energies = krylov_energy_thresholded(H, S, eps) 
	results = {
		"chi": chi,
		"d": ds,
		"energies": energies
	}
	with open(f"data/hubbard_random_chi{chi}_output.json", "w") as f:
		json.dump(results, f)


def main():
	U = 8
	Lx, Ly = 4, 2
	chidmrg = 16

	# generate and save DMRG data for initial state
	E, psi = get_gnd_hubbard(Lx, Ly, chidmrg, U=8, psi_init=None)
	with open(f'data/gnd_chi{chidmrg}.p', 'wb') as handle:
		pickle.dump([E, psi], handle)
	print("Done making DMRG data")

	# generate a few bond dimensions to time evolve with
	# def bond_dims(n):
	# 	return [n + k*n//4 for k in range(5)]
	# chis = bond_dims(chidmrg) + bond_dims(2*chidmrg)[1:]
	# print("Time evolution bond dimensions: ", chis)

	# Make a random state for Krylov
	chi_tdvp = chidmrg
	model_ref = HubbardSquare2D({'Lx': Lx, 'Ly': Ly, 'U':U, 'mu':0})
	sites = [model_ref.lat.site(i) for i in range(Lx * Ly)]
	# psi_random = MPS.from_desired_bond_dimension(sites, chidmrg)
	psi_random = MPS.from_random_unitary_evolution(sites, chi_tdvp, ["up"] * len(sites))

	eps = 1e-8
	dt = 1e-3
	d = 10
	T = d * dt
	random_krylov(psi, Lx, Ly, chi_tdvp, U, T, dt, eps)

if __name__ == '__main__':
	main()
