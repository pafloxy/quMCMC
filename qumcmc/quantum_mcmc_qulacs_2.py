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


### note: (13th June 2023)
# in qulacs, the single qubit rotation operator is defined as exp( + i THETA/2 X/Y/Z)
# similarly, multi qubit rotation operator is defined as exp (+i bla bla bla /2)

def fn_qckt_problem_half(J:np.array,h, num_spins:int, 
                        gamma:float, alpha:float, delta_time=0.8) -> QuantumCircuit:
    ''' 
    Create a quantum circuit for time evolution under scaled problem hamiltonian
    h1 = (1-gamma) * alpha * H_prob
    where H_prob=sum_{j=1}^{n}[-(h_j*Z_j)] + sum_{j>k=1}^{n} [-J_{ij} * Z_{j} * Z_{k}]
    '''
    qc_problem_hamiltonian_half=QuantumCircuit(num_spins)

    avg_interactions = ( np.mean(np.abs(h)), np.mean(np.abs(J))  )
    epsilon = 10**(-3)
    if avg_interactions[0] <= epsilon and avg_interactions[1] <= epsilon: 
        return qc_problem_hamiltonian_half
    
    # 2- qubit term 
    pauli_z_index=[3,3]# (Z tensor Z)
    ### 
    theta_array_2qubit= (2*-1*(1-gamma)*alpha*delta_time)*J #
    for j in range(0,num_spins-1):
        for k in range(j+1, num_spins):
            target_qubit_list=[num_spins-1-j,num_spins-1-k]
            angle= -1*theta_array_2qubit[j,k]# since qulacs convention is +1 in the rotation operator
            qc_problem_hamiltonian_half.add_multi_Pauli_rotation_gate(index_list=target_qubit_list,
                                                pauli_ids=pauli_z_index,
                                                angle=angle)
    # single qubit terms
    ### 
    theta_array_1qubit=(2*-1*(1-gamma)*alpha*delta_time)*np.array(h)
    #target_qubit_list_1qubit=list(range(num_spins-1,-1,-1))
    for j in range(0,num_spins):
        target_qubit=num_spins-1-j
        angle= -1*theta_array_1qubit[j]
        qc_problem_hamiltonian_half.add_RZ_gate(index=target_qubit,
                            angle=angle)
    
    return qc_problem_hamiltonian_half

def fn_qckt_X_mixer(num_spins:int, gamma:float, delta_time:float, 
                    mixer_type = ['random', 1],
                    ):

    '''
    mixer_type = ['random', pauli wt. of mixer] -> All possible mixers of fixed pauli weight will be used.
                 ['custom', [SET OF MIXERS] ] -> All mixers specified in the [SET OF MIXERS] is used
    '''
    qc_evolution_under_mixer=QuantumCircuit(num_spins)
    
    qubit_indices=list(range(0,num_spins))
    
    # if pauli_weight_single_term_mixer==1:
    #     target_qubits_list=list(range(num_spins-1,-1,-1))
    #     angle=-1*2*gamma*delta_time# additional -1 factor since in qulacs convention is +1 in the rotation operator
    #     for j in range(0,num_spins):
    #         #qc_evolution_under_mixer.add_RX_gate(index=target_qubits_list[j],angle=angle)
    #         qc_evolution_under_mixer.add_RX_gate(index=target_qubits_list[j],
    #                                                 angle=angle)

    # elif pauli_weight_single_term_mixer!=1:
    
    if mixer_type[0] == 'custom':
        possible_qubit_combinations = mixer_type[1]
    elif mixer_type[0] == 'random':
         #len(pauli_id_single_term_in_mixer)
        possible_qubit_combinations= [list(i) for i in combinations(qubit_indices, mixer_type[1]) ]  # in ascending order

    # print("possible qubit combination")
    # print(possible_qubit_combinations)

    # create the circuit
    for i in range(0,len(possible_qubit_combinations)):
        
        target_qubits_list= possible_qubit_combinations[i]
        pauli_id_single_term_in_mixer = [1]*len(target_qubits_list)
        angle= -1*2*gamma* delta_time # additional -1 factor since in qulacs convention is +1 in the rotation operator
        
        qc_evolution_under_mixer.add_multi_Pauli_rotation_gate(index_list=target_qubits_list,
                                                        pauli_ids=pauli_id_single_term_in_mixer,
                                                        angle=angle)
    
    return qc_evolution_under_mixer

# def fn_qckt_X_mixer_custom(num_spins:int, gamma:float, delta_time:float, 
#                     mixers: list): ##TODO
    
#     """ 
#     mixers: a list of mixers, where each mixer should be Pauli Operator described 
#             as the string sequence in the Class PauliOperator.
#             For eg.: 'X 0 Y 1 X 2' 
#     """

#     qc_evolution_under_mixer=QuantumCircuit(num_spins)
#     #
#     qubit_indices=list(range(0,num_spins))

#     for mixer in mixers :
#         pass
    
#     return qc_evolution_under_mixer

####
def trotter(num_spins:int, qckt_1:QuantumCircuit, 
                            qckt_2:QuantumCircuit, 
                            num_trotter_steps:int) -> QuantumCircuit:
    qc_combine=QuantumCircuit(num_spins)
    for _ in range(0,num_trotter_steps-1):
        qc_combine.merge_circuit(qckt_2)
        qc_combine.merge_circuit(qckt_1)
    qc_combine.merge_circuit(qckt_2)# this is first order trotterisation
    return qc_combine

####
def run_qmcmc_quantum_ckt(
        state_s:str,
        model: IsingEnergyFunction,
        alpha:float,num_spins:int,
        gamma_range=(0.2,0.6),
        mixer_type = ['random', 1],delta_time=0.8) -> str:
    
    h=model.get_h
    J=model.get_J

 

    time=np.random.choice(list(range(2, 12)))
    gamma=np.round(np.random.uniform(low= min(gamma_range), high = max(gamma_range) ), decimals=6)
    num_trotter_steps=int(np.floor((time / delta_time)))

    #create the circuit
    # initialise the quantum ckt
    qc_for_mcmc=initialise_qc(n_spins=num_spins, 
                                bitstring=state_s)
    qc_problem_half=fn_qckt_problem_half(J=J,h=h,num_spins=num_spins,
                                gamma=gamma, alpha=alpha, delta_time=delta_time)
    qc_mixer=fn_qckt_X_mixer(num_spins=num_spins,gamma=gamma,delta_time=delta_time,
                            mixer_type= mixer_type)
    qc_time_evol=trotter(num_spins=num_spins,qckt_1=qc_problem_half,
                            qckt_2=qc_mixer,
                            num_trotter_steps=num_trotter_steps)
    qc_for_mcmc.merge_circuit(qc_time_evol)

    # run the q ckt
    q_state=QuantumState(qubit_count=num_spins)
    q_state.set_zero_state()
    qc_for_mcmc.update_quantum_state(q_state)
    state_obtained=q_state.sampling(sampling_count=1)[0]
    state_obtained_binary=f"{state_obtained:0{num_spins}b}"
    
    return state_obtained_binary

def quantum_enhanced_mcmc_2(
        n_hops:int,
        model:IsingEnergyFunction,
        mismatched_model = [False, None],
        initial_state:Optional[str]=None,
        temperature:float=1,
        gamma_range:Union[Tuple, List[Tuple]]=(0.2,0.6),
        delta_time=0.8, 
        mixer= [ [['random', 1]], []],
        # pauli_weight_x_mixer:int=1,
        verbose:bool=False,
        name:str = "Q-MCMC"):
    """
    ARGS:
    ----
    Nhops: Number of time you want to run mcmc
    model: The model to be sampled from
    mismathced_model: [False, mismathced_model] If True it uses the mismatched_model for the circuit construction step, 
                        If False, `model` is used both fo circuit construction and energy-comparison    
    mixer: list
        Specifies the type fo mixer configurations and their corresponding probability used during MCMC

        [ [mixer_type1 , mixer_type2 , mixer_type3, . .], [p1, p2, p3, . .] ] }
        

        'mixer_type':
            ['random', pauli wt. of mixer] -> All possible mixers of fixed pauli weight will be used.
            ['custom', [SET OF MIXERS] ] -> All mixers specified in the [SET OF MIXERS] is used
        

    RETURNS:
    -------
    Last 'return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it!
    """
    
    num_spins=model.num_spins

    if initial_state is None:
        initial_state = MCMCState(get_random_state(num_spins), accepted=True)
    else:
        initial_state = MCMCState(initial_state, accepted=True)
    current_state: MCMCState = initial_state
    
    energy_s = model.get_energy(current_state.bitstring)

    if verbose: print("starting with: ", current_state.bitstring, "with energy:", energy_s)

    mcmc_chain = MCMCChain([current_state], name= name +': '+str(mixer))

    # if mixer[0] == 'fixed'  :
    #     for _ in tqdm(range(0, n_hops), desc='runnning quantum MCMC steps . ..', disable= not verbose ):
    #         # get s_prime
    #         s_prime=run_qmcmc_quantum_ckt(state_s=current_state.bitstring,
    #                                         model=model,
    #                                         alpha=model.alpha, num_spins=num_spins,
    #                                         gamma_range=gamma_range,
    #                                         mixer_type= mixer[1] ,
    #                                         delta_time=delta_time
    #                                         )
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



    for _ in tqdm(range(0, n_hops), desc='runnning quantum MCMC steps . ..', disable= not verbose ):
        # get s_prime
        if len(mixer[0]) == 1 :
            mixer_type = mixer[0][0]
        else:
            mixer_type_index = np.random.choice(range(len(mixer[0])), p= mixer[1] )
            mixer_type = mixer[0][mixer_type_index]  
            # print(f"Mixer type is: {mixer_type}")
            gamma = gamma_range
            if type(gamma_range)==list:
                gamma = gamma_range[mixer_type_index]### seems to be working fine
            # print(f"gamma is: {gamma}")
        
        if mismatched_model[0]: 
            model_for_circuit = mismatched_model[1]
        else:
            model_for_circuit = model

        s_prime=run_qmcmc_quantum_ckt(state_s=current_state.bitstring,
                                        model = model_for_circuit,
                                        alpha = model.alpha, num_spins = num_spins,
                                        gamma_range = gamma, #gamma_range,
                                        mixer_type = mixer_type ,
                                        delta_time = delta_time
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











