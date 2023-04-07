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


# def fn_qc_h1(num_spins: int, gamma, alpha, h:list, delta_time=0.8) -> QuantumCircuit :
#     """
#     Create a Quantum Circuit for time-evolution under
#     hamiltonain H1 (described in the paper) 
#     H1= -(1-gamma)*alpha*sum_{j=1}^{n}[(h_j*Z_j)] + gamma *sum_{j=1}^{n}[(X_j)] 

#     ARGS:
#     ----
#     num_spins: number of spins in the model
#     gamma: float
#     alpha: float
#     h: list of field at each site
#     delta_time: total evolution time time/num_trotter_steps
#     """
#     a=gamma
#     b_list=list( ((gamma-1)*(alpha))* np.array(h))
#     qc_h1 = QuantumCircuit(num_spins)
#     for j in range(0, num_spins):
#         unitary_gate=DenseMatrix(index=num_spins-1-j,
#                         matrix=np.round(expm(-1j*delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)
#                         )# this will change accordingly.
#         qc_h1.add_gate(unitary_gate)
#     return qc_h1

def fn_qc_h1(num_spins: int, gamma, alpha, h:list,delta_time=0.8, single_qubit_mixer=True) -> QuantumCircuit :
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
    a=gamma
    b_list = ((gamma-1)*alpha)* np.array(h)
    qc_h1 = QuantumCircuit(num_spins)
    if single_qubit_mixer:
        for j in range(0, num_spins):
            # unitary_gate=DenseMatrix(index=num_spins-1-j,
            #                 matrix=np.round(expm(-1j*delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)
            #                 )# this will change accordingly.
            unitary_gate=DenseMatrix(index=num_spins-1-j,
                            matrix=np.round(expm(-1j*delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)
                            )# this will change accordingly.
            qc_h1.add_gate(unitary_gate)
    else:# added by Neel
        for j in range(0,num_spins):
            matrix_to_exponentiate=b_list[j]*Z(2).get_matrix()
            unitary_gate=DenseMatrix(index=num_spins-1-j,
                                    matrix=np.round(expm(-1j*delta_time*matrix_to_exponentiate),decimals=6)
                                    )
            qc_h1.add_gate(unitary_gate)
        # for j in range(0,num_spins-1):
        #     # unitary_gate=DenseMatrix(index=num_spins-1-j,
        #     #                 matrix=np.round(expm(-1j*delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)
        #     #                 )# this will change accordingly.
        #     matrix_to_exponentiate=a*Pauli([num_spins-1-j,num_spins-2-j],list_Paulis).get_matrix()+b_list[j]*merge([Z(num_spins-2-j),Identity(num_spins-1-j)]).get_matrix()
        #     unitary_gate=DenseMatrix([num_spins-1-j,num_spins-2-j],
        #                     np.round(expm(-1j*delta_time*(matrix_to_exponentiate)),decimals=6)
        #                     )# this will change accordingly.
        #     qc_h1.add_gate(unitary_gate)
        # unitary_gate=DenseMatrix(index=0,
        #         matrix=np.round(expm(-1j*delta_time*(b_list[num_spins-1]*Z(2).get_matrix())),decimals=6)
        #         )# this will change accordingly.
        # qc_h1.add_gate(unitary_gate)

    return qc_h1




def fn_qc_h2(J:np.array, alpha:float, gamma:float, delta_time, single_qubit_mixer=True,pauli_index_list:list=[1,1]) -> QuantumCircuit :
    """
    # updated version.
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
    upper_triag_without_diag=np.triu(J,k=1)
    theta_array=(-2*(1-gamma)*alpha*delta_time)*upper_triag_without_diag
    pauli_z_index=[3,3]## Z tensor Z
    for j in range(0,num_spins-1):
        for k in range(j+1,num_spins):
            #print("j,k is:",(j,k))
            target_list=[num_spins-1-j,num_spins-1-k]#num_spins-1-j,num_spins-1-(j+1)
            angle=theta_array[j,k]
            qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,pauli_ids=pauli_z_index,angle=angle)
    if not(single_qubit_mixer):
        pauli_x_index=pauli_index_list
        indices=list(range(0,num_spins))
        r=len(pauli_x_index)
        all_poss_combn_asc_order=list(combinations(indices,r))
        for i in range(0,len(all_poss_combn_asc_order)):
            target_list=list(all_poss_combn_asc_order[i])
            angle=1
            qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,
                                                        pauli_ids=pauli_x_index, 
                                                        angle=angle)
    return qc_for_evol_h2


def trottered_qc_for_transition(num_spins: int, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int) -> QuantumCircuit:
    """ Returns a trotter circuit (evolution_under_h2 X evolution_under_h1)^(r-1) (evolution under h1)"""
    qc_combine=QuantumCircuit(num_spins)
    for _ in range(0,num_trotter_steps-1):
        qc_combine.merge_circuit(qc_h1)
        qc_combine.merge_circuit(qc_h2)
    qc_combine.merge_circuit(qc_h1)
    return qc_combine


def combine_2_qc(init_qc: QuantumCircuit, trottered_qc: QuantumCircuit) -> QuantumCircuit:
    """ Function to combine 2 quantum ckts of compatible size.
        In this project, it is used to combine initialised quantum ckt and quant ckt meant for time evolution
    """
    num_spins=init_qc.get_qubit_count()
    qc_merge=QuantumCircuit(num_spins)
    qc_merge.merge_circuit(init_qc)
    qc_merge.merge_circuit(trottered_qc)
    return qc_merge


################################################################################################
##  QUANTUM MARKOV CHAIN CONSTRUCTION ##
################################################################################################

def run_qc_quantum_step(
    qc_initialised_to_s: QuantumCircuit, model: IsingEnergyFunction, alpha, n_spins: int,
    single_qubit_mixer=True, pauli_index_list=[1,1]) -> str:

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

    # init_qc=initialise_qc(n_spins=n_spins, bitstring='1'*n_spins)
    gamma = np.round(np.random.uniform(0, 0.01), decimals=6)
    time = np.random.choice(list(range(2, 12)))  # earlier I had [2,20]
    delta_time = 0.8 
    num_trotter_steps = int(np.floor((time / delta_time)))
    # qc_evol_h1 = fn_qc_h1(n_spins, gamma, alpha, h, delta_time)# this will change accordingly
    qc_evol_h1 = fn_qc_h1(n_spins, gamma, alpha, h, delta_time, 
                        single_qubit_mixer=single_qubit_mixer)# newly added by Neel!
    qc_evol_h2 = fn_qc_h2(J, alpha, gamma, delta_time=delta_time, 
                        single_qubit_mixer=single_qubit_mixer,
                        pauli_index_list=pauli_index_list)
    trotter_ckt = trottered_qc_for_transition(
        n_spins, qc_evol_h1, qc_evol_h2, num_trotter_steps=num_trotter_steps
    )
    qc_for_mcmc = combine_2_qc(qc_initialised_to_s, trotter_ckt)# i can get rid of this!
    # run the circuit
    q_state=QuantumState(qubit_count=n_spins)
    q_state.set_zero_state()
    qc_for_mcmc.update_quantum_state(q_state)
    state_obtained=q_state.sampling(sampling_count=1)[0]
    state_obtained_binary=f"{state_obtained:0{n_spins}b}"
    return state_obtained_binary


def quantum_enhanced_mcmc(
    n_hops: int,
    model: IsingEnergyFunction,
    # alpha,
    initial_state: Optional[str] = None,
    temperature=1,
    verbose:bool= False,single_qubit_mixer=True,pauli_index_list=[1,1]
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
    Last 'return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it!
    
    """
    num_spins = model.num_spins

    if initial_state is None:
        initial_state = MCMCState(get_random_state(num_spins), accepted=True)
    else:
        initial_state = MCMCState(initial_state, accepted=True)
    
    current_state: MCMCState = initial_state
    energy_s = model.get_energy(current_state.bitstring)
    if verbose: print("starting with: ", current_state.bitstring, "with energy:", energy_s)

    mcmc_chain = MCMCChain([current_state])

    # print(mcmc_chain)
    for _ in tqdm(range(0, n_hops), desc='runnning quantum MCMC steps . ..', disable= not verbose ):
        # get sprime
        qc_s = initialise_qc(n_spins= model.num_spins, bitstring=current_state.bitstring)
        s_prime = run_qc_quantum_step(
            qc_initialised_to_s=qc_s, model=model, alpha=model.alpha, n_spins= model.num_spins,
            single_qubit_mixer=single_qubit_mixer, pauli_index_list=pauli_index_list
        )
        
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




### JAX not working in windows

# ###########################################################################################
# ## IMPORTS ##
# ###########################################################################################
# import numpy as np
# from typing import Optional
# from tqdm import tqdm
# from collections import Counter
# from .basic_utils import qsm, states, MCMCChain, MCMCState
# # from .prob_dist import *
# from .energy_models import IsingEnergyFunction
# from .classical_mcmc_routines import test_accept, get_random_state
# # from qiskit import (
# #     QuantumCircuit,
# #     QuantumRegister,
# #     ClassicalRegister,
# #     execute,
# # )
# # from qiskit.extensions import UnitaryGate, XGate, ZGate, HamiltonianGate

# # qulacs imports
# from qulacs import QuantumState, QuantumCircuit
# from qulacsvis import circuit_drawer
# from scipy.linalg import expm
# from qulacs.gate import DenseMatrix
# from qulacs.gate import X, Y, Z  

# import jax
# ################################################################################################
# ##  QUANTUM CIRCUIT CONSTRUCTION ##
# ################################################################################################


# def initialise_qc(n_spins: int, bitstring: str) -> QuantumCircuit :
#     """
#     Initialises a quantum circuit with n_spins number of qubits in a state defined by "bitstring"    
#     """
#     qc_in=QuantumCircuit(qubit_count=n_spins)
#     len_str_in = len(bitstring)
#     assert len_str_in==qc_in.get_qubit_count(), "len(bitstring) should be equal to number_of_qubits/spins"

#     for i in range(0,len(bitstring)):
#         if bitstring[i]=="1":
#             qc_in.add_X_gate(len_str_in - 1 - i)
#     return qc_in


# def fn_qc_h1(num_spins: int, gamma, alpha, h:list, delta_time=0.8) -> QuantumCircuit :
#     """
#     Create a Quantum Circuit for time-evolution under
#     hamiltonain H1 (described in the paper) 

#     ARGS:
#     ----
#     num_spins: number of spins in the model
#     gamma: float
#     alpha: float
#     h: list of field at each site
#     delta_time: total evolution time time/num_trotter_steps
#     """
#     a=gamma
#     b_list = ((gamma-1)*alpha)* np.array(h)
#     qc_h1 = QuantumCircuit(num_spins)
#     # for j in range(0, num_spins):
#     #     unitary_gate=DenseMatrix(index=num_spins-1-j,
#     #                     matrix=np.round(
#     #                         # expm(-1j*delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),
#     #                         jaxfunc(a, b_list[j], delta_time),
#     #                         decimals=6)
#     #                     )
#     matrices = np.array(jax_expm_vec(a, b_list, delta_time))
#     for j in range(0, num_spins):
#         unitary_gate=DenseMatrix(index=num_spins-1-j,
#                         matrix=matrices[j]
#                         )
#         qc_h1.add_gate(unitary_gate)
#     return qc_h1


# @jax.jit
# def jax_expm(a, b, delta_time):
#     return jax.scipy.linalg.expm(-1j*delta_time*(a*X(2).get_matrix()+b*Z(2).get_matrix()))

# jax_expm_vec = jax.jit(jax.vmap(jax_expm, (None, 0, None)))


# def fn_qc_h2(J:np.array, alpha:float, gamma:float, delta_time) -> QuantumCircuit :
#     """
#     # updated version.
#     Create a Quantum Circuit for time-evolution under
#     hamiltonain H2 (described in the paper)
#     ARGS:
#     ----
#     J: interaction matrix, interaction between different spins
#     gamma: float
#     alpha: float
#     delta_time: (default=0.8, as suggested in the paper)total evolution time time/num_trotter_steps
#     """
#     num_spins=np.shape(J)[0]
#     qc_for_evol_h2=QuantumCircuit(num_spins)
#     # calculating theta_jk
#     upper_triag_without_diag=np.triu(J,k=1)
#     theta_array=(-2*(1-gamma)*alpha*delta_time)*upper_triag_without_diag
#     pauli_z_index=[3,3]## Z tensor Z
#     for j in range(0,num_spins-1):
#         for k in range(j+1,num_spins):
#             #print("j,k is:",(j,k))
#             target_list=[num_spins-1-j,num_spins-1-k]#num_spins-1-j,num_spins-1-(j+1)
#             angle=theta_array[j,k]
#             qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,pauli_ids=pauli_z_index,angle=angle)
#     return qc_for_evol_h2


# def trottered_qc_for_transition(num_spins: int, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int) -> QuantumCircuit:
#     """ Returns a trotter circuit (evolution_under_h2 X evolution_under_h1)^(r-1) (evolution under h1)"""
#     qc_combine=QuantumCircuit(num_spins)
#     for _ in range(0,num_trotter_steps-1):
#         qc_combine.merge_circuit(qc_h1)
#         qc_combine.merge_circuit(qc_h2)
#     qc_combine.merge_circuit(qc_h1)
#     return qc_combine


# def combine_2_qc(init_qc: QuantumCircuit, trottered_qc: QuantumCircuit) -> QuantumCircuit:
#     """ Function to combine 2 quantum ckts of compatible size.
#         In this project, it is used to combine initialised quantum ckt and quant ckt meant for time evolution
#     """
#     num_spins=init_qc.get_qubit_count()
#     qc_merge=QuantumCircuit(num_spins)
#     qc_merge.merge_circuit(init_qc)
#     qc_merge.merge_circuit(trottered_qc)
#     return qc_merge


# ################################################################################################
# ##  QUANTUM MARKOV CHAIN CONSTRUCTION ##
# ################################################################################################

# def run_qc_quantum_step(
#     qc_initialised_to_s: QuantumCircuit, model: IsingEnergyFunction, alpha, n_spins: int
# ) -> str:

#     """
#     Takes in a qc initialized to some state "s". After performing unitary evolution U=exp(-iHt)
#     , circuit is measured once. Function returns the bitstring s', the measured state .

#     ARGS:
#     ----
#     qc_initialised_to_s:
#     model:
#     alpha:
#     n_spins:
    
#     """

#     h = model.get_h
#     J = model.get_J

#     # init_qc=initialise_qc(n_spins=n_spins, bitstring='1'*n_spins)
#     gamma = np.round(np.random.uniform(0.25, 0.6), decimals=2)
#     time = np.random.choice(list(range(2, 12)))  # earlier I had [2,20]
#     delta_time = 0.8 
#     num_trotter_steps = int(np.floor((time / delta_time)))
#     qc_evol_h1 = fn_qc_h1(n_spins, gamma, alpha, h, delta_time)
#     qc_evol_h2 = fn_qc_h2(J, alpha, gamma, delta_time=delta_time)
#     trotter_ckt = trottered_qc_for_transition(
#         n_spins, qc_evol_h1, qc_evol_h2, num_trotter_steps=num_trotter_steps
#     )
#     qc_for_mcmc = combine_2_qc(qc_initialised_to_s, trotter_ckt)# i can get rid of this!
#     # run the circuit
#     q_state=QuantumState(qubit_count=n_spins)
#     q_state.set_zero_state()
#     qc_for_mcmc.update_quantum_state(q_state)
#     state_obtained=q_state.sampling(sampling_count=1)[0]
#     state_obtained_binary=f"{state_obtained:0{n_spins}b}"
#     return state_obtained_binary


# def quantum_enhanced_mcmc(
#     n_hops: int,
#     model: IsingEnergyFunction,
#     # alpha,
#     initial_state: Optional[str] = None,
#     temperature=1,
#     verbose:bool= False
# ):
#     """
#     version 0.2
    
#     ARGS:
#     ----
#     Nhops: Number of time you want to run mcmc
#     model:
#     return_last_n_states:
#     return_both:
#     temp:

#     RETURNS:
#     -------
#     Last 'return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it!
    
#     """
#     num_spins = model.num_spins

#     if initial_state is None:
#         initial_state = MCMCState(get_random_state(num_spins), accepted=True)
#     else:
#         initial_state = MCMCState(initial_state, accepted=True)
    
#     current_state: MCMCState = initial_state
#     energy_s = model.get_energy(current_state.bitstring)
#     if verbose: print("starting with: ", current_state.bitstring, "with energy:", energy_s)

#     mcmc_chain = MCMCChain([current_state])

#     # print(mcmc_chain)
#     for _ in tqdm(range(0, n_hops), desc='runnning quantum MCMC steps . ..', disable= not verbose ):
#         # get sprime
#         qc_s = initialise_qc(n_spins= model.num_spins, bitstring=current_state.bitstring)
#         s_prime = run_qc_quantum_step(
#             qc_initialised_to_s=qc_s, model=model, alpha=model.alpha, n_spins= model.num_spins
#         )
        
#         if len(s_prime) == model.num_spins :
#             # accept/reject s_prime
#             energy_sprime = model.get_energy(s_prime)
#             accepted = test_accept(
#                 energy_s, energy_sprime, temperature=temperature
#             )
#             mcmc_chain.add_state(MCMCState(s_prime, accepted))
#             if accepted:
#                 current_state = mcmc_chain.current_state
#                 energy_s = model.get_energy(current_state.bitstring)
#         else: pass

#     return mcmc_chain 