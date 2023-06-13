##########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
from typing import Optional
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
from qulacs.gate import X, Y, Z  , Pauli, Identity, merge

from itertools import combinations
import random
################################################################################################
##  QUANTUM CIRCUIT CONSTRUCTION ##
################################################################################################


def initialise_qc(n_spins: int, bitstring: str) -> QuantumCircuit :
    """
    Initialises a quantum circuit with n_spins number of qubits in a state defined by "bitstring"    
    """
    qc_in=QuantumCircuit(qubit_count=n_spins)
    len_str_in = len(bitstring)
    assert len_str_in==qc_in.get_qubit_count(), "len(bitstring) should be equal to number_of_qubits/spins"

    for i in range(0,len(bitstring)):
        if bitstring[i]=="1":
            qc_in.add_X_gate(len_str_in - 1 - i)
    return qc_in


def fn_qc_h1(num_spins: int, gamma, alpha, h:list, delta_time=0.8) -> QuantumCircuit :
    """
    Create a Quantum Circuit for time-evolution under
    hamiltonain H1 (described in the paper) 
    H1= -(1-gamma)*alpha*sum_{j=1}^{n}[(h_j*Z_j)] + gamma *sum_{j=1}^{n}[(X_j)] 

    ARGS:
    ----
    num_spins: number of spins in the model
    gamma: float
    alpha: float
    h: list of field at each site
    delta_time: total evolution time time/num_trotter_steps
    """
    b_list=list( (-1*(1-gamma)*(alpha))* np.array(h))
    qc_h1 = QuantumCircuit(num_spins)
    for j in range(0, num_spins):
        unitary_gate=DenseMatrix(index=num_spins-1-j,
                        matrix=np.round(expm(-1j*delta_time*(gamma*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)
                        )
        qc_h1.add_gate(unitary_gate)
    return qc_h1

def fn_qc_h2(J:np.array, alpha:float, gamma:float, 
                delta_time) -> QuantumCircuit :
    """
    Create a Quantum Circuit for time-evolution under
    hamiltonain H2 (described in the paper)
    ARGS:
    ----
    J: interaction matrix, interaction between different spins
    gamma: float
    alpha: float
    delta_time: (default=0.8, as suggested in the paper)total evolution time time/num_trotter_steps
    """
    num_spins=np.shape(J)[0]
    qc_for_evol_h2=QuantumCircuit(num_spins)
    # calculating theta_jk
    theta_array = (2*-1*(1-gamma)*alpha*delta_time)*J # *2 is intentional because rotation operator has a factor of 2 in Denominator in its phase.
    pauli_z_index=[3,3]## Z tensor Z
    for j in range(0,num_spins-1):
        for k in range(j+1,num_spins):
            #print("j,k is:",(j,k))
            target_list=[num_spins-1-j,num_spins-1-k]#num_spins-1-j,num_spins-1-(j+1)
            angle=-1*theta_array[j,k]# additional -1 factor since in qulacs convention is +1 in the rotation operator
            qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,
                                                        pauli_ids=pauli_z_index,angle=angle)
    return qc_for_evol_h2

def trottered_qc_for_transition(num_spins: int, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int) -> QuantumCircuit:
    """ Returns a trotter circuit (evolution_under_h2 X evolution_under_h1)^(r-1) (evolution under h1)"""
    qc_combine=QuantumCircuit(num_spins)
    for _ in range(0,num_trotter_steps-1):
        qc_combine.merge_circuit(qc_h1)
        qc_combine.merge_circuit(qc_h2)
    qc_combine.merge_circuit(qc_h1)# added by me just now. june 10th 2023.
    return qc_combine


# def combine_2_qc(init_qc: QuantumCircuit, trottered_qc: QuantumCircuit) -> QuantumCircuit:
#     """ Function to combine 2 quantum ckts of compatible size.
#         In this project, it is used to combine initialised quantum ckt and quant ckt meant for time evolution
#     """
#     num_spins=init_qc.get_qubit_count()
#     qc_merge=QuantumCircuit(num_spins)
#     qc_merge.merge_circuit(init_qc)
#     qc_merge.merge_circuit(trottered_qc)
#     return qc_merge


################################################################################################
##  QUANTUM MARKOV CHAIN CONSTRUCTION ##
################################################################################################

def run_qc_quantum_step(current_state_s: str, 
                        model: IsingEnergyFunction, alpha, n_spins: int, 
                        gamma_range= (0.2, 0.6)) -> str:

    """
    Takes in a qc initialized to some state "s". After performing unitary evolution U=exp(-iHt)
    , circuit is measured once. Function returns the bitstring s', the measured state .

    ARGS:
    ----
    qc_initialised_to_s:
    model:
    alpha:
    n_spins:
    
    """

    h = model.get_h
    J = model.get_J

    gamma = np.round(np.random.uniform(low= min(gamma_range), high = max(gamma_range) ), decimals=6)
    time = np.random.choice(list(range(2, 12)))  # earlier I had [2,20]
    delta_time = 0.8 
    num_trotter_steps = int(np.floor((time / delta_time)))
    qc_for_mcmc=initialise_qc(n_spins=n_spins, bitstring=current_state_s)
    qc_evol_h1 = fn_qc_h1(n_spins, gamma, alpha, h, delta_time)
    qc_evol_h2 = fn_qc_h2(J, alpha, gamma, delta_time=delta_time)
    qc_time_evol = trottered_qc_for_transition(n_spins, qc_evol_h1, qc_evol_h2, 
                                                num_trotter_steps=num_trotter_steps)
    qc_for_mcmc.merge_circuit(qc_time_evol)
    # run the circuit
    q_state= QuantumState(qubit_count=n_spins)
    q_state.set_zero_state()
    qc_for_mcmc.update_quantum_state(q_state)
    state_obtained=q_state.sampling(sampling_count=1)[0]
    state_obtained_binary=f"{state_obtained:0{n_spins}b}"
    return state_obtained_binary


def quantum_enhanced_mcmc(
    n_hops: int,
    model: IsingEnergyFunction,
    initial_state: Optional[str] = None,
    temperature=1,
    gamma_range = (0.2, 0.6),
    verbose:bool= False,
    name:str = "quMCMC"
):
    """
    version 0.2
    
    ARGS:
    ----
    Nhops: Number of time you want to run mcmc
    model:
    return_last_n_states:
    return_both:
    temp   

    Returns:
    MCMC chain.
    """
    num_spins = model.num_spins

    if initial_state is None:
        initial_state = MCMCState(get_random_state(num_spins), accepted=True)
    else:
        initial_state = MCMCState(initial_state, accepted=True)
    
    current_state: MCMCState = initial_state
    energy_s = model.get_energy(current_state.bitstring)
    if verbose: print("starting with: ", current_state.bitstring, "with energy:", energy_s)

    mcmc_chain = MCMCChain([current_state], name= name)

    for _ in tqdm(range(0, n_hops), desc='runnning quantum MCMC steps . ..', disable= not verbose ):
        # get sprime
        #qc_s = initialise_qc(n_spins= model.num_spins, bitstring=current_state.bitstring)
        s_prime = run_qc_quantum_step(
            current_state_s=current_state.bitstring, 
            model=model, alpha=model.alpha, n_spins= model.num_spins, 
            gamma_range= gamma_range)
        
        if len(s_prime) == model.num_spins :
            # accept/reject s_prime
            energy_sprime = model.get_energy(s_prime)
            accepted = test_accept(
                energy_s, energy_sprime, temperature=temperature
            )
            mcmc_chain.add_state(MCMCState(s_prime, accepted))
            if accepted:
                current_state = mcmc_chain.current_state
                energy_s = model.get_energy(current_state.bitstring)
        else: pass

    return mcmc_chain 