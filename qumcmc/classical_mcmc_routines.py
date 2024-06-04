###########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
# from .basic_utils import *
# from .prob_dist import *
from .energy_models import IsingEnergyFunction
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import Counter
from .basic_utils import MCMCChain, MCMCState, xor_strings, random_bstr, get_random_state
from .classical_mixers import ClassicalMixer, UniformProposals

from abc import ABC, abstractmethod
from typing import List
###########################################################################################
## CLASSICAL MCMC ROUTINES ##
###########################################################################################


def test_accept(
    energy_s: float, energy_sprime: float, temperature: float = 1.
) -> MCMCState:
    """
    Accepts the state "sprime" with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) )
    and s_init with probability 1-A.
    """
    delta_energy = energy_sprime - energy_s  # E(s')-E(s)
    exp_factor = np.exp(-delta_energy / temperature)
    acceptance = min(
        1, exp_factor
    )  # for both QC case as well as uniform random strategy, the transition matrix Pij is symmetric!

    return acceptance > np.random.rand()

def classical_mcmc(
    n_hops: int,
    model: IsingEnergyFunction ,
    proposition_method: ClassicalMixer = UniformProposals(1),
    initial_state: Optional[str] = None,
    temperature: float = 1.,
    verbose:bool= False,
    name = 'cl-mcmc'
    # num_flips:int = 1
):
    """
    ARGS:
    -----
    n_hops: Number of time you want to run mcmc
    model: 
    initial_state:
    temperature:
    methods : Choose proposition strategy -> 
            [ [ method1, method2, method3 ] , [p1, p2, p3] ] ; i.e method1 is opted with probability p1 and so on. 

            method ->  ['uniform'] / ['local', num-flips]
    
    RETURNS:
    --------
    Last 'dict_count_return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it!
    
    """
    num_spins = model.num_spins

    # assert isinstance(method, str); assert method in {'uniform', 'local'},  ("Unkown method specified, choose betwen ('uniform', 'local') ")

    if initial_state is None : 
        initial_state = MCMCState(get_random_state(num_spins), accepted=True)
    else:
        initial_state = MCMCState(initial_state, accepted=True)
    
    current_state: MCMCState = initial_state
    energy_s = model.get_energy(current_state.bitstring)
    if verbose : print("starting with: ", current_state.bitstring, "with energy:", energy_s)

    mcmc_chain = MCMCChain([current_state], name= name )


    for _ in tqdm(range(0, n_hops), desc= 'running MCMC steps ...', disable= not verbose):
        # get sprime
        # if len(proposition_method[0]) == 1 : 
        #     method = proposition_method[0][0][0]; 
        #     if method != 'uniform' : num_flips = proposition_method[0][0][1]
        # else : 
        #     p_i = np.random.choice(range(len(proposition_method[0])), p = proposition_method[1])
        #     method = proposition_method[0][p_i][0] 
        #     if method != 'uniform' : num_flips = proposition_method[0][p_i][1]

        # if method == 'uniform' :
        #     s_prime = get_random_state(num_spins)
        # elif method == 'local' :
        #     rbstr = random_bstr(num_spins, num_flips)
        #     s_prime = xor_strings(current_state.bitstring, rbstr)
        s_prime = proposition_method.propose_transition(current_state.bitstring)

        # accept/reject s_prime
        energy_sprime = model.get_energy(s_prime)   # to make this scalable, I think you need to calculate energy ratios.
        accepted = test_accept(
            energy_s, energy_sprime, temperature=temperature
        )
        mcmc_chain.add_state(MCMCState(s_prime, accepted))
        if accepted:
            current_state = mcmc_chain.current_state
            energy_s = model.get_energy(current_state.bitstring)
        
    return mcmc_chain

# class ClassicalMixer(ABC):
    
#     def __init__(self, num_spins: int) -> None:
#         self.num_spins = num_spins
#         # self.current_state = current_state
#         self._precompute_properties()

#     @abstractmethod
#     def propose_transition(self): ...

#     def _precompute_properties(self):
#         pass

# class UniformProposals(ClassicalMixer):

#     def _init_(self, num_spins:int) -> None: 
#         super().__init__(num_spins)

#     def propose_transition(self):
#         return get_random_state(self.num_spins)

# class FixedWeightProposals(ClassicalMixer):

#     def __init__(self, num_spins: int, bodyness: int, current_state: str ) -> None:
#         self.bodyness = bodyness
#         self.current_state = current_state

#         assert self.bodyness <= self.num_spins , "Incorrect"
        
#         super().__init__(num_spins)
        
#     def propose_transition(self):
#         rbstr = random_bstr(self.num_spins, self.bodyness)
#         return xor_strings(self.current_state, rbstr)

# class CustomProposals(ClassicalMixer):

#     def __init__(self, num_spins: int, flip_pattern: List[int], current_state: str) -> None:
#         self.flip_pattern =flip_pattern
#         self.current_state = current_state
#         super().__init__(num_spins)

#     def propose_transition(self):
#         rbstr = ''
#         __ = [ '1' if i in self.flip_pattern else '0' for i in range(self.num_spins)]
#         for _ in __ : rbstr += _ 
#         return xor_strings(self.current_state, rbstr)
    
# class CombineProposals(ClassicalMixer):

#     def __init__(self, proposal_methods: List[ClassicalMixer], probabilities: List[float], current_state: str) -> None:
#         # super().__init__(num_spins) 
#         self.num_spins = proposal_methods[0].num_spins
#         assert all(
#             self.num_spins == pm.num_spins for pm in proposal_methods
#         ), "Mixers don't have the same number of qubits"
#         assert len(probabilities) == len(
#             proposal_methods
#         ), "Length of list of mixers and probabilities is not equal" 

#         self.probabilities = np.array(probabilities) / sum(probabilities)
#         self.proposal_methods = proposal_methods
#         self.current_state = current_state

#         def propose_transition(self):
#             proposal_method = np.random.choice(self.proposal_methods, p = self.probabilities)
#             return proposal_method.propose_transition()
#             # return 