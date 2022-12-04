###########################################################################################
## IMPORTS ##
###########################################################################################

from .basic_utils import *
from .prob_dist import *
from .energy_models import *

###########################################################################################
## CLASSICAL MCMC ROUTINES ##
###########################################################################################

def classical_transition(num_spins: int) -> str:
    """
    Returns s' , obtained via uniform transition rule!
    """
    num_elems = 2 ** (num_spins)
    next_state = np.random.randint(
        0, num_elems
    )  # since upper limit is exclusive and lower limit is inclusive
    bin_next_state = f"{next_state:0{num_spins}b}"
    return bin_next_state

def classical_loop_accepting_state(
    s_init: str, s_prime: str, energy_s: float, energy_sprime: float, temp=1
) -> str:
    """
    Accepts the state "sprime" with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) )
    and s_init with probability 1-A.
    """
    delta_energy = energy_sprime - energy_s  # E(s')-E(s)
    exp_factor = np.exp(-delta_energy / temp)
    acceptance = min(
        1, exp_factor
    )  # for both QC case as well as uniform random strategy, the transition matrix Pij is symmetric!
    # coin_flip=np.random.choice([True, False], p=[acceptance, 1-acceptance])
    new_state = s_init
    if acceptance >= np.random.uniform(0, 1):
        new_state = s_prime
    return new_state

def classical_mcmc(
    N_hops: int,
    num_spins: int,
    initial_state: str,
    num_elems: int,# i dont think it is being used anywhere. 
    model,
    return_last_n_states=500,
    return_additional_lists=False,
    temp=1,
):
    """
    ARGS:
    -----
    Nhops: Number of time you want to run mcmc
    num_spins: number of spins
    num_elems: 2**(num_spins)
    model:
    return_last_n_states: (int) Number of states in the end of the M.Chain you want to consider for prob distn (default value is last 500)
    return_both (default=False): If set to True, in addition to dict_count_return_lst_n_states, also returns 2 lists:
                                "list_after_transition: list of states s' obtained after transition step s->s' " and
                                "list_state_mchain_is_in: list of states markov chain was in".
    RETURNS:
    --------
    Last 'dict_count_return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it!
    
    """
    states_obt = []
    # current_state=f'{np.random.randint(0,num_elems):0{num_spins}b}'# bin_next_state=f'{next_state:0{num_spins}b}'
    current_state = initial_state
    print("starting with: ", current_state)
    states_obt.append(current_state)

    ## initialiiise observables
    # observable_dict = dict([ (elem, []) for elem in observables ])
    list_after_transition = []
    list_state_mchain_is_in = []
    poss_states=states(num_spins=num_spins)# list of all possible states

    for i in tqdm(range(0, N_hops), desc= 'running MCMC steps ...'):
        # get sprime
        s_prime = classical_transition(num_spins)
        list_after_transition.append(s_prime)
        # accept/reject s_prime
        energy_s = model.get_energy(current_state)
        energy_sprime = model.get_energy(s_prime)
        next_state = classical_loop_accepting_state(
            current_state, s_prime, energy_s, energy_sprime, temp=temp
        )
        current_state = next_state
        list_state_mchain_is_in.append(current_state)
        states_obt.append(current_state)
        # WE DON;T NEED TO DO THIS! # reinitiate
        # qc_s=initialise_qc(n_spins=num_spins, bitstring=current_state)

    # returns dictionary of occurences for last "return_last_n_states" states
    dict_count_return_last_n_states=dict(zip(poss_states,[0]*(len(poss_states))))
    dict_count_return_last_n_states.update(dict(Counter(states_obt[-return_last_n_states:])))
    if return_additional_lists:
        to_return = (
            dict_count_return_last_n_states,
            list_after_transition,
            list_state_mchain_is_in,
        )
    else:
        to_return = dict_count_return_last_n_states

    return to_return
