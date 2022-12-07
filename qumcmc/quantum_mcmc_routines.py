###########################################################################################
## IMPORTS ##
###########################################################################################

from .basic_utils import *
from .prob_dist import *
from .energy_models import *
from .classical_mcmc_routines import *

################################################################################################
##  QUANTUM CIRCUIT CONSTRUCTION ##
################################################################################################


def initialise_qc(n_spins: int, bitstring: str) -> QuantumCircuit :
    """
    Initialises a quantum circuit with n_spins number of qubits in a state defined by "bitstring"
    (## NOTE : Qiskit's indexing convention for qubits (order of tensor product) is different from the conventional textbook one!)
    
    """

    spins = QuantumRegister(n_spins, name="spin")
    creg_final = ClassicalRegister(n_spins, name="creg_f")
    qc_in = QuantumCircuit(spins, creg_final)

    len_str_in = len(bitstring)
    assert len_str_in == len(
        qc_in.qubits
    ), "len(bitstring) should be equal to number_of_qubits/spins"

    # print("qc_in.qubits: ", qc_in.qubits)
    where_x_gate = [
        qc_in.qubits[len_str_in - 1 - i]
        for i in range(0, len(bitstring))
        if bitstring[i] == "1"
    ]
    if len(where_x_gate) != 0:
        qc_in.x(where_x_gate)
    return qc_in


def fn_qc_h1(num_spins: int, gamma, alpha, h, delta_time) -> QuantumCircuit :
    """
    Create a Quantum Circuit for time-evolution under
    hamiltonain H1 (described in the paper) 

    ARGS:
    ----
    num_spins: number of spins in the model
    gamma: float
    alpha: float
    h: list of field at each site
    delta_time: total evolution time time/num_trotter_steps


    """
    a = gamma
    # print("a:",a)
    b_list = [-(1 - gamma) * alpha * hj for hj in h]
    list_unitaries = [
        UnitaryGate(
            HamiltonianGate(
                a * XGate().to_matrix() + b_list[j] * ZGate().to_matrix(),
                time=delta_time,
            ).to_matrix(),
            label=f"exp(-ia{j}X+b{j}Z)",
        )
        for j in range(0, num_spins)
    ]
    qc = QuantumCircuit(num_spins)
    for j in range(0, num_spins):
        qc.append(list_unitaries[j], [num_spins - 1 - j])
    qc.barrier()
    # print("qc is:"); print(qc.draw())
    return qc


def fn_qc_h2(J:np.array, alpha:float, gamma:float, delta_time=0.8) -> QuantumCircuit :
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
    num_spins = np.shape(J)[0]
    qc_for_evol_h2 = QuantumCircuit(num_spins)
    theta_list = [
        -2 * J[j, j + 1] * (1 - gamma) * alpha * delta_time
        for j in range(0, num_spins - 1)
    ]
    for j in range(0, num_spins - 1):
        qc_for_evol_h2.rzz(
            theta_list[j], qubit1=num_spins - 1 - j, qubit2=num_spins - 1 - (j + 1)
        )
    # print("qc for fn_qc_h2 is:"); print(qc_for_evol_h2.draw())
    return qc_for_evol_h2


def trottered_qc_for_transition(num_spins: int, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int) -> QuantumCircuit:
    """ Returns a trotter circuit (evolution_under_h2 X evolution_under_h1)^(r-1) (evolution under h1)"""
    qc_combine = QuantumCircuit(num_spins)
    for i in range(0, num_trotter_steps - 1):
        qc_combine = qc_combine.compose(qc_h1)
        qc_combine = qc_combine.compose(qc_h2)
        qc_combine.barrier()
    qc_combine = qc_combine.compose(qc_h1)
    # print("trotter ckt:"); print(qc_combine.draw())
    return qc_combine


def combine_2_qc(init_qc: QuantumCircuit, trottered_qc: QuantumCircuit) -> QuantumCircuit:
    """ Function to combine 2 quantum ckts of compatible size.
        In this project, it is used to combine initialised quantum ckt and quant ckt meant for time evolution
    """
    num_spins = len(init_qc.qubits)
    qc = QuantumCircuit(num_spins, num_spins)
    qc = qc.compose(init_qc)
    qc.barrier()
    qc = qc.compose(trottered_qc)
    return qc

######## classical loop acceptance state #####
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
################################################################################################
##  QUANTUM MARKOV CHAIN CONSTRUCTION ##
################################################################################################

def run_qc_quantum_step(
    qc_initialised_to_s: QuantumCircuit, model: IsingEnergyFunction, alpha, n_spins: int
) -> str:

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

    h = model.get_h# and not model.get_h() anymore
    J = model.get_J# and not model.get_J() anymore

    # init_qc=initialise_qc(n_spins=n_spins, bitstring='1'*n_spins)
    gamma = np.round(np.random.uniform(0.25, 0.6), decimals=2)
    time = np.random.choice(list(range(2, 12)))  # earlier I had [2,20]
    delta_time = 0.8
    num_trotter_steps = int(np.floor((time / delta_time)))
    # print(f"gamma:{gamma}, time: {time}, delta_time: {delta_time}, num_trotter_steps:{num_trotter_steps}")
    # print(f"num troter steps: {num_trotter_steps}")
    qc_evol_h1 = fn_qc_h1(n_spins, gamma, alpha, h, delta_time)
    qc_evol_h2 = fn_qc_h2(J, alpha, gamma, delta_time=delta_time)
    trotter_ckt = trottered_qc_for_transition(
        n_spins, qc_evol_h1, qc_evol_h2, num_trotter_steps=num_trotter_steps
    )
    qc_for_mcmc = combine_2_qc(qc_initialised_to_s, trotter_ckt)

    # run the circuit
    num_shots = 1
    quantum_registers_for_spins = qc_for_mcmc.qregs[0]
    classical_register = qc_for_mcmc.cregs[0]
    qc_for_mcmc.measure(quantum_registers_for_spins, classical_register)
    # print("qc_for_mcmc: ")
    # print( qc_for_mcmc.draw())
    state_obtained_dict = (
        execute(qc_for_mcmc, shots=num_shots, backend=qsm).result().get_counts()
    )
    state_obtained = list(state_obtained_dict.keys())[
        0
    ]  # since there is only one element
    return state_obtained


def quantum_enhanced_mcmc(
    N_hops: int,
    # num_spins: int,
    # num_elems: int,
    model: IsingEnergyFunction,
    # alpha,
    initial_state: Union[None, str] = None,
    return_last_n_states=500,
    return_additional_lists=False,
    temp=1,
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
    states_obt = []
    all_configs = [f"{k:0{model.num_spins}b}" for k in range(0, 2 ** (model.num_spins))]
    if initial_state == None : 
        initial_state = np.random.choice(all_configs)
    print("starting with: ", initial_state)

    ## initialise quantum circuit to current_state
    qc_s = initialise_qc(n_spins= model.num_spins, bitstring=initial_state)
    current_state = initial_state
    states_obt.append(current_state)
    ## intialise observables
    list_after_transition = []
    list_state_mchain_is_in = []
    poss_states=states(num_spins=model.num_spins)

    for i in tqdm(range(0, N_hops), desc='runnning quantum MCMC steps . ..' ):
        # print("i: ", i)
        # get sprime
        s_prime = run_qc_quantum_step(
            qc_initialised_to_s=qc_s, model=model, alpha=model.alpha, n_spins= model.num_spins
        )
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
        ## reinitiate
        qc_s = initialise_qc(n_spins= model.num_spins, bitstring=current_state)

    # dict_count_return_last_n_states = Counter(
    #     states[-return_last_n_states:]
    # )  # dictionary of occurences for last "return_last_n_states" states
    #
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