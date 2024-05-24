##########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
from typing import Optional, Union, List, Tuple
from tqdm import tqdm
from collections import Counter
from .basic_utils import qsm, states, MCMCChain, MCMCState

# from .prob_dist import *
from .energy_models import IsingEnergyFunction
from .classical_mcmc_routines import test_accept, get_random_state

# qulacs imports
from qulacs import QuantumState, QuantumCircuit
from qulacsvis import circuit_drawer
from scipy.linalg import expm
from qulacs.gate import DenseMatrix
from qulacs.gate import X, Y, Z, Pauli, Identity, merge

from itertools import combinations
import random

from .mixers import Mixer

GammaType = Union[float, Tuple[float, float]]

################################################################################################
##  QUANTUM CIRCUIT CONSTRUCTION ##
################################################################################################


def initialise_qc(n_spins: int, bitstring: str) -> QuantumCircuit:
    """
    Initialises a quantum circuit with n_spins number of qubits in a state defined by "bitstring"
    """
    qc_in = QuantumCircuit(qubit_count=n_spins)
    len_str_in = len(bitstring)
    assert (
        len_str_in == qc_in.get_qubit_count()
    ), "len(bitstring) should be equal to number_of_qubits/spins"

    for i in range(0, len(bitstring)):
        if bitstring[i] == "1":
            qc_in.add_X_gate(len_str_in - 1 - i)
    return qc_in


### note: (13th June 2023)
# in qulacs, the single qubit rotation operator is defined as exp( + i THETA/2 X/Y/Z)
# similarly, multi qubit rotation operator is defined as exp (+i bla bla bla /2)


def fn_qckt_problem_half(
    J: np.array, h, num_spins: int, gamma: float, alpha: float, delta_time=0.8
) -> QuantumCircuit:
    """
    Create a quantum circuit for time evolution under scaled problem hamiltonian
    h1 = (1-gamma) * alpha * H_prob
    where H_prob=sum_{j=1}^{n}[-(h_j*Z_j)] + sum_{j>k=1}^{n} [-J_{ij} * Z_{j} * Z_{k}]
    """
    qc_problem_hamiltonian_half = QuantumCircuit(num_spins)

    avg_interactions = (np.mean(np.abs(h)), np.mean(np.abs(J)))
    epsilon = 10 ** (-3)
    if avg_interactions[0] <= epsilon and avg_interactions[1] <= epsilon:
        return qc_problem_hamiltonian_half

    # 2- qubit term
    pauli_z_index = [3, 3]  # (Z tensor Z)
    ###
    theta_array_2qubit = (2 * -1 * (1 - gamma) * alpha * delta_time) * J  #
    for j in range(0, num_spins - 1):
        for k in range(j + 1, num_spins):
            target_qubit_list = [num_spins - 1 - j, num_spins - 1 - k]
            angle = (
                -1 * theta_array_2qubit[j, k]
            )  # since qulacs convention is +1 in the rotation operator
            qc_problem_hamiltonian_half.add_multi_Pauli_rotation_gate(
                index_list=target_qubit_list, pauli_ids=pauli_z_index, angle=angle
            )
    # single qubit terms
    ###
    theta_array_1qubit = (2 * -1 * (1 - gamma) * alpha * delta_time) * np.array(h)

    for j in range(0, num_spins):
        target_qubit = num_spins - 1 - j
        angle = -1 * theta_array_1qubit[j]
        qc_problem_hamiltonian_half.add_RZ_gate(index=target_qubit, angle=angle)

    return qc_problem_hamiltonian_half


####
def trotter(
    num_spins: int,
    qckt_1: QuantumCircuit,
    qckt_2: QuantumCircuit,
    num_trotter_steps: int,
) -> QuantumCircuit:
    qc_combine = QuantumCircuit(num_spins)
    for _ in range(0, num_trotter_steps - 1):
        qc_combine.merge_circuit(qckt_2)
        qc_combine.merge_circuit(qckt_1)
    qc_combine.merge_circuit(qckt_2)  # this is first order trotterisation
    return qc_combine


####
def run_qmcmc_quantum_ckt(
    state_s: str,
    model: IsingEnergyFunction,
    mixer: Mixer,
    gammafunc: callable,
    alpha: float,
    num_spins: int,
    delta_time: float,
):

    h = model.circuit_h
    J = model.circuit_J

    gamma = gammafunc()

    time = np.random.choice(list(range(2, 12)))

    num_trotter_steps = int(np.floor((time / delta_time)))

    # create the circuit
    # initialise the quantum ckt
    qc_for_mcmc = initialise_qc(n_spins=num_spins, bitstring=state_s)
    qc_problem_half = fn_qckt_problem_half(
        J=J,
        h=h,
        num_spins=num_spins,
        gamma=gamma,
        alpha=alpha,
        delta_time=delta_time,
    )
    qc_mixer = mixer.get_mixer_circuit(gamma, delta_time)
    qc_time_evol = trotter(
        num_spins=num_spins,
        qckt_1=qc_problem_half,
        qckt_2=qc_mixer,
        num_trotter_steps=num_trotter_steps,
    )
    qc_for_mcmc.merge_circuit(qc_time_evol)

    # run the q ckt
    q_state = QuantumState(qubit_count=num_spins)
    q_state.set_zero_state()
    qc_for_mcmc.update_quantum_state(q_state)
    state_obtained = q_state.sampling(sampling_count=1)[0]
    state_obtained_binary = f"{state_obtained:0{num_spins}b}"

    return state_obtained_binary


def quantum_enhanced_mcmc(
    n_hops: int,
    model: IsingEnergyFunction,
    mixer: Mixer,
    gamma: GammaType,
    initial_state: Optional[str] = None,
    temperature: float = 1,
    delta_time=0.8,
    verbose: bool = False,
    name: str = "Q-MCMC",
):
    """
    ARGS:
    ----
    Nhops: Number of time you want to run mcmc
    model: The model to be sampled from
    mixer: Mixer
    RETURNS:
    -------
    Last 'return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it!
    """

    num_spins = model.num_spins

    if isinstance(gamma, Union[float, int]):
        gammafunc = lambda: gamma
    else:
        gammafunc = lambda: np.random.uniform(*gamma)

    if initial_state is None:
        initial_state = MCMCState(get_random_state(num_spins), accepted=True)
    else:
        initial_state = MCMCState(initial_state, accepted=True)
    current_state: MCMCState = initial_state

    energy_s = model.get_energy(current_state.bitstring)

    if verbose:
        print("starting with: ", current_state.bitstring, "with energy:", energy_s)

    mcmc_chain = MCMCChain([current_state], name=name + ": " + str(mixer))

    for _ in tqdm(
        range(0, n_hops), desc="runnning quantum MCMC steps . ..", disable=not verbose
    ):

        s_prime = run_qmcmc_quantum_ckt(
            state_s=current_state.bitstring,
            model=model,
            mixer=mixer,
            gammafunc=gammafunc,
            alpha=model.alpha,
            num_spins=num_spins,
            delta_time=delta_time,
        )
        if len(s_prime) == model.num_spins:
            # accept/reject s_prime
            energy_sprime = model.get_energy(s_prime)

            accepted = test_accept(energy_s, energy_sprime, temperature=temperature)
            mcmc_chain.add_state(MCMCState(s_prime, accepted))
            if accepted:
                current_state = mcmc_chain.current_state
                energy_s = model.get_energy(current_state.bitstring)

        else:
            pass

    return mcmc_chain
