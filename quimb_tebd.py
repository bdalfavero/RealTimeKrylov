from typing import List, Tuple, Union
import numpy as np
import scipy.linalg as la
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct
from quimb.tensor.circuit import CircuitMPS
import qiskit
from qiskit.quantum_info import SparsePauliOp
import cirq
from qiskit.qasm2 import dumps
import openfermion as of
from openfermionpyscf import run_pyscf

def pauli_string_to_mpo(pstring: cirq.PauliString, qs: List[cirq.Qid]) -> MatrixProductOperator:
    """Convert a Pauli string to a matrix product operator."""

    # Make a list of matrices for each operator in the string.
    ps_dense = pstring.dense(qs)
    matrices: List[np.ndarray] = []
    for pauli_int in ps_dense.pauli_mask:
        if pauli_int == 0:
            matrices.append(np.eye(2))
        elif pauli_int == 1:
            matrices.append(cirq.unitary(cirq.X))
        elif pauli_int == 2:
            matrices.append(cirq.unitary(cirq.Y))
        else: # pauli_int == 3
            matrices.append(cirq.unitary(cirq.Z))
    # Convert the matrices into tensors. We have a bond dim chi=1 for a Pauli string MPO.
    tensors: List[np.ndarray] = []
    for i, m in enumerate(matrices):
        if i == 0:
            tensors.append(m.reshape((2, 2, 1)))
        elif i == len(matrices) - 1:
            tensors.append(m.reshape((1, 2, 2)))
        else:
            tensors.append(m.reshape((1, 2, 2, 1)))
    return pstring.coefficient * MatrixProductOperator(tensors, shape="ludr")


def pauli_sum_to_mpo(psum: cirq.PauliSum, qs: List[cirq.Qid], max_bond: int) -> MatrixProductOperator:
    """Convert a Pauli sum to an MPO."""

    for i, p in enumerate(psum):
        if i == 0:
            mpo = pauli_string_to_mpo(p, qs)
        else:
            mpo += pauli_string_to_mpo(p, qs)
            tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)
    return mpo


def cirq_pauli_sum_to_qiskit_pauli_op(pauli_sum: cirq.PauliSum) -> SparsePauliOp:
    """Returns a qiskit.SparsePauliOp representation of the cirq.PauliSum."""

    cirq_pauli_to_str = {cirq.X: "X", cirq.Y: "Y", cirq.Z: "Z"}

    qubits = pauli_sum.qubits
    terms = []
    coeffs = []
    for term in pauli_sum:
        string = ""
        for qubit in qubits:
            if qubit not in term:
                string += "I"
            else:
                string += cirq_pauli_to_str[term[qubit]]
        terms.append(string)
        assert np.isclose(term.coefficient.imag, 0.0, atol=1e-7)
        coeffs.append(term.coefficient.real)
    return SparsePauliOp(terms, coeffs)


def mps_to_vector(mps: MatrixProductState) -> np.ndarray:
    """Convert an MPS into a normal vector. This assumes each index is a string
    followed by a number, e.g. three indices 'k0, k1, k2'."""

    def _idx_to_int(idx: str) -> int:
        digits = [c for c in idx if c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        if len(digits) == 0:
            raise ValueError(f"Index {str} has no digits in it.")
        return int(''.join(digits))
    
    # Contract the MPS into a tensor, then sort the indices. Convert that to a vector.
    contracted_tensor = mps.contract()
    sorted_inds = sorted(contracted_tensor.inds, key=_idx_to_int)
    print(f"sorted_inds = {sorted_inds}")
    contracted_tensor.transpose(*sorted_inds, inplace=True)
    tensor_data = contracted_tensor.data
    return tensor_data.reshape((tensor_data.size,))


def total_number_qubit_operator(n_orbitals: int, use_jw=True) -> of.QubitOperator:
    """Get a Pauli sum representing the total number operator.
    
    Arugments:
    n_orbitals - Number of orbitals in the system.
    use_jw - Whether to use Jordan-Wigner. Otherwise, use Bravyi-Kitaev.
    
    Returns:
    A QubitOperator representation of the total number operator."""

    total_number = of.FermionOperator.zero()
    for i in range(n_orbitals):
        total_number += of.FermionOperator(f"{i}^ {i}", 1.0)
    if use_jw:
        total_number_qubit = of.transforms.jordan_wigner(total_number)
    else:
        total_number_qubit = of.transforms.bravyi_kitaev(total_number)
    return total_number_qubit


def mpo_mps_exepctation(mpo: MatrixProductOperator, mps: MatrixProductState) -> complex:
    """Get the expectation of an operator given the state."""

    mpo_times_mps = mpo.apply(mps)
    return mps.H @ mpo_times_mps


def get_drmg_ground_state(
    ham_mpo: MatrixProductOperator, n_elec: int, chi: int, alpha: float=1.0
) -> Tuple[MatrixProductState, float, float]:
    """Get the ground state with n_elec electrons using DMRG with bond dimension chi.
    
    Arguments:
    ham_mpo - MPO representation of the Hamiltonian.
    n_elec - Number of electrons in the system.
    chi - Bond dimension for DMRG.
    alpha - A factor to ensure total number. We find the ground state of
    H' = H + alpha * (N - n_elec)**2.
    
    Returns:
    ground_state - Output of DMRG.
    energy - The energy of the ground state we found.
    n - Expectation of the number of electrons in the ground state."""

    n_orbitals = len(ham_mpo.tensors)
    assert n_elec <= n_orbitals
    total_number_of = total_number_qubit_operator(n_orbitals)
    total_number_cirq = of.transforms.qubit_operator_to_pauli_sum(total_number_of)
    occupation_term = alpha * (total_number_cirq - n_elec) ** 2
    qs = cirq.LineQubit.range(of.utils.count_qubits(total_number_of))
    occupation_mpo = pauli_sum_to_mpo(occupation_term, qs, 100)
    total_number_mpo = pauli_sum_to_mpo(total_number_cirq, qs, 100)
    ham_mpo_augmented = ham_mpo + occupation_mpo

    dmrg = qtn.DMRG(ham_mpo_augmented, chi)
    converged = dmrg.solve()
    if not converged:
        print("DMRG did not converge!")
    psi = dmrg.state
    energy = mpo_mps_exepctation(ham_mpo, psi)
    number = mpo_mps_exepctation(total_number_mpo, psi)
    return (psi, energy.real, number.real)


def trotter_circuit_from_psum(hamiltonian: cirq.PauliSum, t: float, steps: int) -> qiskit.QuantumCircuit:
    """Convert a PauliSum into a Trotter circuit with Paulihedral."""

    dt = t / float(steps)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(hamiltonian)
    ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
    nq = len(hamiltonian.qubits)
    ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
    for _ in range(steps):
        ev_ckt_qiskit.append(ev_gate, range(nq))
    return ev_ckt_qiskit


def get_evolved_states(
    evolution_circuit: qiskit.QuantumCircuit,
    reference_mps: MatrixProductState,
    d: int,
    max_circuit_bond: int,
    backend_callback=None
) -> List[MatrixProductState]:
    """Get a list of d evolved states."""

    states: List[MatrixProductState] = []
    for i in range(d):
        if i == 0:
            evolved_mps = reference_mps.copy()
        else:
            # Make a circuit with d repetitions of the evolution circuit.
            # nq = evolution_circuit.num_qubits
            # total_circuit = qiskit.QuantumCircuit(nq)
            # for _ in range(i):
            #     total_circuit = total_circuit.compose(evolution_circuit)
            # Convert the circuit to quimb format.
            # qasm_str = dumps(total_circuit)
            # if backend_callback is not None:
            #     circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
            #         qasm_str, psi0=reference_mps, max_bond=max_circuit_bond, progbar=False,
            #         to_backend=backend_callback
            #     )
            # else:
            #     circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
            #         qasm_str, psi0=reference_mps, max_bond=max_circuit_bond, progbar=False
            #     )
            qasm_str = dumps(evolution_circuit)
            if backend_callback is not None:
                circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                    qasm_str, psi0=evolved_mps, max_bond=max_circuit_bond, progbar=False,
                    to_backend=backend_callback
                )
            else:
                circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                    qasm_str, psi0=evolved_mps, max_bond=max_circuit_bond, progbar=False
                )
            evolved_mps = circuit_mps.psi
        evolved_mps.normalize()
        states.append(evolved_mps.copy())
    return states


def exact_evolved_states(
    evolution_circuit: qiskit.QuantumCircuit,
    reference_state: np.ndarray,
    d: int,
) -> List[np.ndarray]:
    """Compute <psi|HU^d|psi> and <psi|U^d|psi> using matrix multiplication"""

    # gate = evolution_circuit.to_gate()
    # u = gate.to_matrix()
    states = []
    u = qiskit.quantum_info.Operator(evolution_circuit).data
    evolved_state = reference_state.copy()
    for _ in range(d):
        states.append(evolved_state.copy())
        evolved_state = u @ evolved_state
    return states


def tebd_matrix_element_and_overlap(
    ham_mpo: MatrixProductOperator,
    evolution_circuit: qiskit.QuantumCircuit,
    reference_mps: MatrixProductState,
    d: int,
    max_circuit_bond: int,
    backend_callback
) -> Tuple[complex, complex]:
    """Compute <psi|HU^d|psi> and <psi|U^d|psi> using TEBD"""

    if d == 0:
        evolved_mps = reference_mps.copy()
    else:
        # Make a circuit with d repetitions of the evolution circuit.
        nq = evolution_circuit.num_qubits
        total_circuit = qiskit.QuantumCircuit(nq)
        for _ in range(d):
            total_circuit = total_circuit.compose(evolution_circuit)
        # Convert the circuit to quimb format.
        qasm_str = dumps(total_circuit)
        if backend_callback is not None:
            circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                qasm_str, psi0=reference_mps, max_bond=max_circuit_bond, progbar=False,
                to_backend=backend_callback
            )
        else:
            circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                qasm_str, psi0=reference_mps, max_bond=max_circuit_bond, progbar=False
            )
        evolved_mps = circuit_mps.psi
    evolved_mps.normalize()
    # Build tensor networks for <psi| U^d |psi> and <psi| H U^d |psi>
    overlap = reference_mps.H @ evolved_mps
    mat_elem = reference_mps.H @ ham_mpo.apply(evolved_mps)
    return (mat_elem, overlap)


def exact_matrix_element_and_overlap(
    ham_matrix: np.ndarray,
    evolution_circuit: qiskit.QuantumCircuit,
    reference_state: np.ndarray,
    d: int,
) -> Tuple[complex, complex]:
    """Compute <psi|HU^d|psi> and <psi|U^d|psi> using matrix multiplication"""

    # gate = evolution_circuit.to_gate()
    # u = gate.to_matrix()
    u = qiskit.quantum_info.Operator(evolution_circuit).data
    evolved_state = reference_state.copy()
    for _ in range(d):
        evolved_state = u @ evolved_state
    overlap = np.vdot(reference_state, evolved_state)
    mat_elem = np.vdot(reference_state, ham_matrix @ evolved_state)
    return (mat_elem, overlap)


def fill_subspace_matrices_toeplitz(
    mat_elems: List[complex], overlaps: List[complex]
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill subspace matrices from the computed matrix elements and overlaps."""

    assert len(mat_elems) == len(overlaps)
    d = len(mat_elems)
    h = np.zeros((d, d), dtype=complex)
    s = np.zeros((d, d), dtype=complex)
    # for i in range(d): # Loop over rows.
    #     for j in range(i+1, d):
    #         h[i, j] = mat_elems[j - i]
    #         s[i, j] = overlaps[j - i]
    # h += h.conj().T
    # s += s.conj().T
    # for i in range(d):
    #     h[i, i] = mat_elems[0]
    #     s[i, i] = overlaps[0]
    for i in range(d):
        for j in range(d):
            if i >= j:
                h[i, j] = mat_elems[i-j].conj()
                s[i, j] = overlaps[i-j].conj()
            else:
                h[i, j] = mat_elems[j-i]
                s[i, j] = overlaps[j-i]
    return h, s


def fill_subspace_matrices_mps(
    states: List[MatrixProductState],
    hamiltonian_mpo: MatrixProductOperator
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill the matrices form the list of states."""

    N = len(states)
    S, H = [np.zeros((N, N), dtype=complex) for _ in range(2)]

    # fill off-diagonal elements with overlaps and hamiltonian expectation values
    for i in range(N):
        for j in range(i+1, N):
            # S[i, j] = states[i].overlap(states[j]) # < vi | vj >
            # H[i, j] = tp.MPOEnvironment(states[i], model_ref.H_MPO, states[j]).full_contraction(0) # < vi | H | vj >
            S[i, j] = states[i].H @ states[j]
            H[i, j] = states[i].H @ hamiltonian_mpo.apply(states[j])
    H += H.conj().T
    S += S.conj().T

    # fill diagonal elements 
    for i in range(N):
        # S[i, i] = states[i].overlap(states[i]).real
        # H[i, i] = model_ref.H_MPO.expectation_value(states[i]).real
        S[i, i] = states[i].H @ states[i]
        H[i, i] = states[i].H @ hamiltonian_mpo.apply(states[i])
    # print("In state-based")
    # print("||H - H^dag|| =", la.norm(H - H.conj().T))
    # print("||S - S^dag|| =", la.norm(S - S.conj().T))
    return (H, S)


def fill_subspace_matrices_vectors(
    states: List[np.ndarray],
    ham: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill the matrices form the list of states."""

    N = len(states)
    S, H = [np.zeros((N, N), dtype=complex) for _ in range(2)]

    # fill off-diagonal elements with overlaps and hamiltonian expectation values
    for i in range(N):
        for j in range(i+1, N):
            # S[i, j] = states[i].H @ states[j]
            # H[i, j] = states[i].H @ hamiltonian_mpo.apply(states[j])
            S[i, j] = np.vdot(states[i], states[j])
            H[i, j] = np.vdot(states[i], ham @ states[j])
    H += H.conj().T
    S += S.conj().T

    # fill diagonal elements 
    for i in range(N):
        # S[i, i] = states[i].H @ states[i]
        # H[i, i] = states[i].H @ hamiltonian_mpo.apply(states[i])
        S[i, i] = np.vdot(states[i], states[i])
        H[i, i] = np.vdot(states[i], ham @ states[i])
    return (H, S)


def threshold_eigenvalues(h: np.ndarray, s: np.ndarray, eps: float, verbose: bool=False) -> Tuple[np.ndarray, np.ndarray, int]:
    """Remove all eigenvalues below a positive threshold eps.
    See Epperly et al. sec. 1.2."""

    # Build a matrix whose columns correspond to the positive eigenvectors of s.
    evals, evecs = la.eigh(s)
    if verbose:
        print("All eigenvalues of S:", evals)
    positive_evals = []
    positive_evecs = []
    num_kept = 0
    for i, ev in enumerate(evals):
        assert abs(ev.imag) < 1e-7
        if ev.real > eps:
            positive_evals.append(ev.real)
            positive_evecs.append(evecs[:, i])
            num_kept += 1
    if verbose:
        print(f"Kept {num_kept} eigenvalues out of {len(evals)}.")
    if num_kept == 0:
        raise RuntimeError(f"No eigenvalues kept. Eigenvalues of S are\n{evals}")
    pos_evec_mat = np.vstack(positive_evecs).T
    # Project h and s into this subspace.
    new_s =  pos_evec_mat.conj().T @ s @ pos_evec_mat
    new_h = pos_evec_mat.conj().T @ h @ pos_evec_mat
    return new_h, new_s, num_kept


def energy_vs_d(
    h: np.ndarray, s: np.ndarray,
    method: str = "threshold", **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Get energy from H, S for each dimension up to the total size d of the subspace."""

    assert h.shape == s.shape
    assert method in ["threshold", "identity"]
    if method == "threshold":
        assert "eps" in kwargs
    if method == "identity":
        assert "eta" in kwargs

    energies = []
    num_kept = []
    for d in range(1, h.shape[0]):
        h_d = h[:d, :d]
        s_d = s[:d, :d]
        if method == "threshold":
            new_h, new_s, nkept = threshold_eigenvalues(h_d, s_d, kwargs["eps"])
        else:
            new_h = h_d.copy()
            new_s = s_d + kwargs["eta"] * np.eye(s_d.shape[0])
            nkept = h_d.shape[0]
        eigvals, eigvecs = la.eig(new_h, new_s)
        i_min = np.argmin(eigvals.real)
        # print(eigvecs[:, i_min].conj().T @ eigvecs[:, i_min])
        energies.append(eigvals[i_min].real)
        num_kept.append(nkept)
    return np.array(energies), np.array(num_kept)