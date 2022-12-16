###################################################################################
## IMPORTS ##
###################################################################################

import numpy as np
from typing import Optional
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass

from .basic_utils import qsm, states, MCMCChain
# from .prob_dist import *
from .energy_models import IsingEnergyFunction
from .classical_mcmc_routines import test_accept, get_random_state
from .quantum_mcmc_routines_qulacs import *



###################################################################################
## REDEFINE DATACLASSES ##
###################################################################################
@dataclass
class MCMCState:
    
    var: str
    fixed: str
    
    accepted: bool = False
    
    def __post_init__(self):
        self.bitstring = self.var + self.fixed
        self.len_var = len(self.var)
        self.len_fixed = len(self.fixed)
        self.len = len(self.bitstring)
    
    def _update_var(self, new_state:str):
        if len(new_state) == self.len_var:
            self.var = new_state
            self.bitstring = self.var + self.fixed
        else : raise ValueError("Updated 'var' should be of len "+str(self.len_var))

###################################################################################
## HELPER SUBROUTINES FOR QUANTUM MCMC ##
###################################################################################  



###################################################################################
## REDEFINE SAMPLING ROUTINES ##
###################################################################################

from typing import Union

@dataclass
class RestrictedSampling:
        
        model : IsingEnergyFunction
        # iterations : int = 10000
        temperature : float = 1.00
        initial_state : Optional[Union[ str, MCMCState]] = None
        

        
                
        def __post_init__(self):
                
                if self.initial_state is None : 
                        self.initial_state = MCMCState(get_random_state(self.model.num_spins), accepted=True)
                elif not isinstance(self.initial_state, MCMCState):
                        self.initial_state = MCMCState(self.initial_state, accepted=True)
                
                self.current_state: MCMCState = self.initial_state
                
                self.mcmc_chain: MCMCChain = MCMCChain([self.current_state])

                self.len_var = self.current_state.len_var; self.len_fixed = self.current_state.len_fixed 
                
                

        def run_classical_mcmc(self, iterations):
                
                energy_s = self.model.get_energy(self.current_state.bitstring)
                print('current state: ', self.current_state)
                for _ in tqdm(range(0, iterations), desc= 'running MCMC steps ...'):
                        # get sprime #
                        s_prime = MCMCState(get_random_state(self.current_state.len_var  ), self.current_state.fixed)
                        print('s_prime:', s_prime)
                        
                        # accept/reject s_prime
                        energy_sprime = self.model.get_energy(s_prime.bitstring)   # to make this scalable, I think you need to calculate energy ratios.
                        accepted = test_accept(
                        energy_s, energy_sprime, temperature=self.temperature
                        )
                        if accepted:
                                s_prime.accepted = accepted
                                self.current_state = s_prime
                                print('current state: ', self.current_state)
                                energy_s = self.model.get_energy(self.current_state.bitstring)
                        
                        self.mcmc_chain.add_state(s_prime)

                return self.mcmc_chain

      
        def _get_quantum_proposition(
            self, qc_initialised_to_s: QuantumCircuit, check_fixed: bool = True, max_checks: int = 100
        ) -> str:

            """
            Takes in a qc initialized to some state "s". After performing unitary evolution U=exp(-iHt)
            , circuit is measured once. Function returns the bitstring s', the measured state .

            ARGS:
            ----
            qc_initialised_to_s:
            model:
            
            """

            h = self.model.get_h
            J = self.model.get_J

            # init_qc=initialise_qc(model.num_spins=model.num_spins, bitstring='1'*model.num_spins)
            gamma = np.round(np.random.uniform(0.25, 0.6), decimals=2)
            time = np.random.choice(list(range(2, 12)))  # earlier I had [2,20]
            delta_time = 0.8 
            num_trotter_steps = int(np.floor((time / delta_time)))
            qc_evol_h1 = fn_qc_h1(self.model.num_spins, gamma, self.model.alpha, h, delta_time)
            qc_evol_h2 = fn_qc_h2(J, self.model.alpha, gamma, delta_time=delta_time)
            trotter_ckt = trottered_qc_for_transition(
                self.model.num_spins, qc_evol_h1, qc_evol_h2, num_trotter_steps=num_trotter_steps
            )
            qc_for_mcmc = combine_2_qc(qc_initialised_to_s, trotter_ckt)# i can get rid of this!
            # run the circuit ##
            q_state=QuantumState(qubit_count=self.model.num_spins)
            q_state.set_zero_state()
            qc_for_mcmc.update_quantum_state(q_state)

            check_fixed_state = lambda bitstr : bitstr[ - self.initial_state.len_fixed: ] == self.initial_state.fixed
            if check_fixed :
                ## repeats sampling untill right fixed state is found ##
                right_sample = False; checks= 0
                while not right_sample and checks < max_checks:
                    state_obtained= q_state.sampling(sampling_count= 1)[0] ; checks+= 1
                    if check_fixed_state( f"{state_obtained:0{self.model.num_spins}b}" ) : right_sample = True
                    
            else :
                state_obtained= q_state.sampling(sampling_count= 1)[0]

            # state_obtained= [f"{state:0{model.num_spins}b}" for state in state_obtained]
            return f"{state_obtained:0{self.model.num_spins}b}"

        def run_quantum_enhanced_mcmc(self, iterations:int , verbose:bool = False):

                energy_s = self.model.get_energy(self.current_state.bitstring)
                if verbose: print('current state: ', self.current_state)
                qc_s = initialise_qc(n_spins= self.model.num_spins, bitstring= self.current_state.bitstring )
                for _ in tqdm(range(0, iterations), desc='runnning quantum MCMC steps . ..' ):
                        
                        # get sprime #
                        qc_s = initialise_qc(n_spins= self.model.num_spins, bitstring=self.current_state.bitstring)
                        s_prime = self._get_quantum_proposition(
                        qc_initialised_to_s=qc_s
                        )
                        s_prime = MCMCState(s_prime[:self.len_var], s_prime[self.len_var:])
                        if verbose: print('s_prime:', s_prime)

                        # accept/reject s_prime
                        energy_sprime = self.model.get_energy(s_prime.bitstring)
                        accepted = test_accept(
                        energy_s, energy_sprime, temperature=self.temperature
                        )
                        if accepted:
                                s_prime.accepted = accepted
                                self.current_state = s_prime
                                print('current state: ', self.current_state)
                                energy_s = self.model.get_energy(self.current_state.bitstring)
                        
                        self.mcmc_chain.add_state(s_prime)
                                

                return self.mcmc_chain 
                

