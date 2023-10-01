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
from .basic_utils import MCMCChain, MCMCState, xor_strings, random_bstr



###########################################################################################
## CLASSICAL MCMC ROUTINES ##
###########################################################################################

def get_random_state(num_spins: int) -> str:
    """
    Returns s' , obtained via uniform transition rule!
    """
    num_elems = 2 ** (num_spins)
    next_state = np.random.randint(
        0, num_elems
    )  # since upper limit is exclusive and lower limit is inclusive
    bin_next_state = f"{next_state:0{num_spins}b}"
    return bin_next_state


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
    initial_state: Optional[str] = None,
    temperature: float = 1.,
    verbose:bool= False,
    proposition_method= [  [['uniform']], [] ],
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
        if len(proposition_method[0]) == 1 : 
            method = proposition_method[0][0][0]; 
            if method != 'uniform' : num_flips = proposition_method[0][0][1]
        else : 
            p_i = np.random.choice(range(len(proposition_method[0])), p = proposition_method[1])
            method = proposition_method[0][p_i][0] 
            if method != 'uniform' : num_flips = proposition_method[0][p_i][1]

        if method == 'uniform' :
            s_prime = get_random_state(num_spins)
        elif method == 'local' :
            rbstr = random_bstr(num_spins, num_flips)
            s_prime = xor_strings(current_state.bitstring, rbstr)

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

