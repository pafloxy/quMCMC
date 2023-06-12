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
# from qiskit import (
#     QuantumCircuit,
#     QuantumRegister,
#     ClassicalRegister,
#     execute,
# )
# from qiskit.extensions import UnitaryGate, XGate, ZGate, HamiltonianGate

# qulacs imports
from qulacs import QuantumState, QuantumCircuit
from qulacs import Observable
from qulacsvis import circuit_drawer
from scipy.linalg import expm
from qulacs.gate import DenseMatrix, SparseMatrix
from qulacs.gate import X, Y, Z  , Pauli, Identity, merge

from itertools import combinations
import random

################################################################################################
##  QUANTUM MARKOV CHAIN CONSTRUCTION ##
################################################################################################

# function for U=exp(-iHt) (a unitary evolution)
# H =(1-gamma)*alpha*H_{prob} + gamma * H_mix
def time_evolution(num_spins:int, hamiltonian:np.array,
                        current_q_state:QuantumState, 
                        time_evol:float):
    '''
    Exact unitary time evolution under the hamiltonian H
    Question: 
    1.  What should be the type of input q_state? 
    2. Should I return a QuantumState object or a simple computational bitstring?
    '''
    #target_list=list(range(num_spins-1,-1,-1))
    target_list=list(range(0,num_spins))
    unitary_mat= np.round(expm(-1j*hamiltonian*time_evol),decimals=6)
    unitary_time_evol_gate=DenseMatrix(target_list,matrix=unitary_mat)
    # update the quantum state
    unitary_time_evol_gate.update_quantum_state(current_q_state)
    # I need to sample something here and return a bitstring
    state_obtained=current_q_state.sampling(sampling_count=1)[0]
    state_obtained_binary=f"{state_obtained:0{num_spins}b}"
    return state_obtained_binary


def create_X_mixer_hamiltonian(num_spins:int,weight_individual_pauli:int):
    mixer_hamiltonian=Observable(num_spins)
    print("type_mixer_hamiltonian:",type(mixer_hamiltonian))
    num_sumterms=num_spins-weight_individual_pauli+1
    list_individual_terms=[("X %d "*weight_individual_pauli) % tuple(range(i,i+weight_individual_pauli)) for i in range(0,num_sumterms)]
    print("list_individual_terms:")
    print(list_individual_terms)
    for i in range(0,num_sumterms):
        mixer_hamiltonian.add_operator(coef=1,string=list_individual_terms[i])
    return mixer_hamiltonian

def quantum_mcmc_exact(
    n_hops: int,
    model: IsingEnergyFunction,
    H_mix: Observable,
    initial_state: Optional[str] = None,
    gamma_range:tuple=(0.2,0.6),# we will be varying this
    temperature=1,
    verbose:bool= False
):
    """
    version 0.2
    
    ARGS:
    ----
    Nhops: Number of time you want to run mcmc
    model:
    return_last_n_states:
    return_both:
    temp:

    RETURNS:
    -------
    mcmc chain. 
    """
    num_spins = model.num_spins

    if initial_state is None:
        initial_state = MCMCState(get_random_state(num_spins), accepted=True)
    else:
        initial_state = MCMCState(initial_state, accepted=True)
    
    current_state: MCMCState = initial_state
    #instantiate a QuantumState object for current_state: current_quantum_state
    current_quantum_state=QuantumState(num_spins)
    current_quantum_state.set_computational_basis(int(current_state.bitstring,2))

    energy_s = model.get_energy(current_state.bitstring)
    if verbose: print("starting with: ", current_state.bitstring, "with energy:", energy_s)
    mcmc_chain = MCMCChain([current_state])

    alpha=model.alpha
    H_prob_obj=model.get_hamiltonian()
    H_prob_array=H_prob_obj.get_matrix().toarray()
    H_mixer_array=H_mix.get_matrix().toarray()
    
    #if I keep gamma fixed
    #gamma=0.5#float(np.round(np.random.uniform(low= min(gamma_range), high = max(gamma_range) ), decimals=6))
    #hamiltonian_mcmc=(1-gamma)*alpha*(-1)*H_prob_array + gamma * H_mixer_array #this needs to come here.
    
    for _ in tqdm(range(0, n_hops), desc='runnning quantum MCMC steps . ..', disable= not verbose ):
        gamma=float(np.round(np.random.uniform(low= min(gamma_range), high = max(gamma_range) ), decimals=6))
        hamiltonian_mcmc=(1-gamma)*alpha*(-1)*H_prob_array + gamma * H_mixer_array #this needs to come here.
        time_evol=int(np.random.choice(list(range(2,12))))
        # get sprime
        s_prime=time_evolution(num_spins=num_spins,
                          hamiltonian=hamiltonian_mcmc,
                          current_q_state=current_quantum_state,
                          time_evol=time_evol)
        #print("sprime is:",s_prime)
        if len(s_prime) == model.num_spins :
            # accept/reject s_prime
            energy_sprime = model.get_energy(s_prime)
            accepted = test_accept(
                energy_s, energy_sprime, temperature=temperature
            )
            mcmc_chain.add_state(MCMCState(s_prime, accepted))
            if accepted:
                current_state = mcmc_chain.current_state
                #re-instantiate current state QuantumState object
                current_quantum_state=QuantumState(num_spins)
                current_quantum_state.set_computational_basis(int(current_state.bitstring,2))
                energy_s = model.get_energy(current_state.bitstring)
        #print("s_current is:",current_state.bitstring)
    return mcmc_chain 