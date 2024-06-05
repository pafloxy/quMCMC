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
from .basic_utils import (
    MCMCChain,
    MCMCState,
    xor_strings,
    random_bstr,
    get_random_state,
)
from .classical_mixers import ClassicalMixer, UniformProposals

from abc import ABC, abstractmethod
from typing import List

###########################################################################################
## CLASSICAL MCMC ROUTINES ##
###########################################################################################


def test_accept(
    energy_s: float, energy_sprime: float, temperature: float = 1.0
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
    model: IsingEnergyFunction,
    proposition_method: ClassicalMixer,
    initial_state: Optional[str] = None,
    temperature: float = 1.0,
    verbose: bool = False,
    name="cl-mcmc",
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

    if initial_state is None:
        initial_state = MCMCState(get_random_state(num_spins), accepted=True)
    else:
        initial_state = MCMCState(initial_state, accepted=True)

    current_state: MCMCState = initial_state
    energy_s = model.get_energy(current_state.bitstring)
    if verbose:
        print("starting with: ", current_state.bitstring, "with energy:", energy_s)

    mcmc_chain = MCMCChain([current_state], name=name)

    for _ in tqdm(range(0, n_hops), desc="running MCMC steps ...", disable=not verbose):

        s_prime = proposition_method.propose_transition(current_state.bitstring)

        # accept/reject s_prime
        energy_sprime = model.get_energy(
            s_prime
        )  # to make this scalable, I think you need to calculate energy ratios.
        accepted = test_accept(energy_s, energy_sprime, temperature=temperature)
        mcmc_chain.add_state(MCMCState(s_prime, accepted))
        if accepted:
            current_state = mcmc_chain.current_state
            energy_s = model.get_energy(current_state.bitstring)

    return mcmc_chain
