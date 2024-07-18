###########################################################################################
## IMPORTS ##
###########################################################################################

## basic imports ##
import numpy as np
import pandas as pd

from tqdm import tqdm

import random
from itertools import permutations
from collections import Counter
from collections_extended import IndexedDict
from typing import Optional, List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt

## qiskit imports ##
# from qiskit import Aer
# from qiskit import (  # , InstructionSet
#     IBMQ,
#     Aer,
#     ClassicalRegister,
#     QuantumCircuit,
#     QuantumRegister,
#     quantum_info,
# )
# from qiskit.algorithms import *
# from qiskit.circuit.library import *
# from qiskit.circuit.library import RXGate, RZGate, RZZGate
# from qiskit.circuit.quantumregister import Qubit
# from qiskit.extensions import HamiltonianGate, UnitaryGate
# from qiskit.quantum_info import Pauli, Statevector, partial_trace, state_fidelity
# from qiskit.utils import QuantumInstance
# from qiskit.visualization import (
#     plot_bloch_multivector,
#     plot_bloch_vector,
#     plot_histogram,
#     plot_state_qsphere,
# )

## qiskit-backends ##
# qsm = Aer.get_backend("qasm_simulator")
# stv = Aer.get_backend("statevector_simulator")
# aer = Aer.get_backend("aer_simulator")


###########################################################################################
## MCMC Chain and States ##
###########################################################################################


@dataclass
class MCMCState:
    bitstring: str
    accepted: bool


@dataclass(init=True)
class MCMCChain:
    def __init__(
        self, states: Optional[List[MCMCState]] = None, name: Optional[str] = "MCMC"
    ):

        self.name = name

        if len(states) is None:
            self._states: List[MCMCState] = []
            self._current_state: MCMCState = None
            self._states_accepted: List[MCMCState] = []
            self.markov_chain: List[str] = []

        else:
            self._states = states
            self._current_state: MCMCState = next(
                (s for s in self._states[::-1] if s.accepted), None
            )
            self._states_accepted: List[MCMCState] = [
                state for state in states if state.accepted
            ]
            self.markov_chain: List[str] = self.get_list_markov_chain()

    def add_state(self, state: MCMCState):
        if state.accepted:
            self._current_state = state
            self._states_accepted.append(state)
        self.markov_chain.append(self._current_state.bitstring)
        self._states.append(state)

    @property
    def states(self):
        return self._states

    @property
    def current_state(self):
        return self._current_state

    @property
    def accepted_states(self) -> List[str]:

        return [state.bitstring for state in self._states_accepted]

    ### added by neel 13-Jan-2023 - edited by Manuel 12-Feb
    def get_list_markov_chain(self) -> List[str]:
        markov_chain_in_state = [self.states[0].bitstring]
        for i in range(1, len(self.states)):
            mcmc_state = self.states[i].bitstring
            whether_accepted = self.states[i].accepted
            if whether_accepted == True:
                markov_chain_in_state.append(mcmc_state)
            else:
                markov_chain_in_state.append(markov_chain_in_state[i - 1])
        self.markov_chain = markov_chain_in_state
        return self.markov_chain

    def get_accepted_dict(self, normalize: bool = False, until_index: int = -1):
        if until_index != -1:
            accepted_states = self.markov_chain[:until_index]
        else:
            accepted_states = self.markov_chain

        if normalize:
            length = len(accepted_states)
            accepted_dict = Counter(
                {s: count / length for s, count in Counter(accepted_states).items()}
            )
        else:
            accepted_dict = Counter(accepted_states)

        return accepted_dict


###########################################################################################
## HELPER FUNCTIONS ##
###########################################################################################


def uncommon_els_2_lists(list_1, list_2):
    return list(set(list_1).symmetric_difference(set(list_2)))


def merge_2_dict(dict1, dict2):
    return {**dict1, **dict2}


def sort_dict_by_keys(dict_in: dict):
    return dict(IndexedDict(sorted(dict_in.items())))


def xor_strings(s1, s2):
    # Ensure strings are of the same length
    assert len(s1) == len(s2), "Strings must be of the same length"
    l = len(s1)
    # Convert each character to binary, perform XOR, then convert back to character
    xor_result = int(s1, 2) ^ int(s2, 2)
    return int_to_binary(xor_result, l)


def states(num_spins: int) -> list:
    """
    Returns all possible binary strings of length n=num_spins

    Args:
    num_spins: n length of the bitstring
    Returns:
    possible_states= list of all possible binary strings of length num_spins
    """
    num_possible_states = 2 ** (num_spins)
    possible_states = [f"{k:0{num_spins}b}" for k in range(0, num_possible_states)]
    return possible_states


def magnetization_of_state(bitstring: str) -> float:
    """
    Args:
    bitstring: for eg: '010'
    Returns:
    magnetization for the given bitstring
    """
    array = np.array(list(bitstring))
    num_times_one = np.count_nonzero(array == "1")
    num_times_zero = len(array) - num_times_one
    magnetization = num_times_one - num_times_zero
    return magnetization


def dict_magnetization_of_all_states(list_all_possible_states: list) -> dict:
    """
    Returns magnetization for all unique states

    Args:
    list_all_possible_states
    Returns:
    dict_magnetization={state(str): magnetization_value}
    """
    list_mag_vals = [
        magnetization_of_state(state) for state in list_all_possible_states
    ]
    dict_magnetization = dict(zip(list_all_possible_states, list_mag_vals))
    # print("dict_magnetization:"); print(dict_magnetization)
    return dict_magnetization


def value_sorted_dict(dict_in, reverse=False):
    """Sort the dictionary in ascending or descending(if reverse=True) order of values"""
    sorted_dict = {
        k: v
        for k, v in sorted(dict_in.items(), key=lambda item: item[1], reverse=reverse)
    }
    return sorted_dict


def value_sorted_dict(dict_in, reverse=False):
    """Sort the dictionary in ascending or descending(if reverse=True) order of values"""
    sorted_dict = {
        k: v
        for k, v in sorted(dict_in.items(), key=lambda item: item[1], reverse=reverse)
    }
    return sorted_dict


## enter samples, get normalised distn
def get_distn(list_of_samples: list) -> dict:
    """
    Returns the dictionary of distn for input list_of_samples
    """
    len_list = len(list_of_samples)
    temp_dict = Counter(list_of_samples)
    temp_prob_list = np.array(list(temp_dict.values())) * (1.0 / len_list)
    dict_to_return = dict(zip(list(temp_dict.keys()), temp_prob_list))
    return dict_to_return


## Average
def avg(dict_probabilities: dict, dict_observable_val_at_states: dict):
    """
    new version:
    Returns average of any observable of interest

    Args:
    dict_probabilities= {state: probability}
    dict_observable_val_at_states={state (same as that of dict_probabilities): observable's value at that state}

    Returns:
    avg
    """
    len_dict = len(dict_probabilities)
    temp_list = [
        dict_probabilities[j] * dict_observable_val_at_states[j]
        for j in (list(dict_probabilities.keys()))
    ]
    avg = np.sum(
        temp_list
    )  # earlier I had np.mean here , which is wrong (obviously! duh!)
    return avg


### function to get running average of magnetization
def running_avg_magnetization(list_states_mcmc: list):
    """
    Returns the running average magnetization

    Args:
    list_states_mcmc= List of states aceepted after each MCMC step
    """
    len_iters_mcmc = len(list_states_mcmc)
    running_avg_mag = {}
    for i in tqdm(range(1, len_iters_mcmc)):
        temp_list = list_states_mcmc[:i]  # [:i]
        temp_prob = get_distn(temp_list)
        dict_mag_states_in_temp_prob = dict_magnetization_of_all_states(temp_list)
        running_avg_mag[i] = avg(temp_prob, dict_mag_states_in_temp_prob)
    return running_avg_mag


def running_avg_magnetization_as_list(list_states_mcmc: list):
    """
    Returns the running average magnetization

    Args:
    list_states_mcmc= List of states aceepted after each MCMC step
    """
    list_of_strings = list_states_mcmc
    list_of_lists = (
        np.array([list(int(s) for s in bitstring) for bitstring in list_of_strings]) * 2
        - 1
    )
    return np.array(
        [
            np.mean(np.sum(list_of_lists, axis=1)[:ii])
            for ii in range(1, len(list_states_mcmc) + 1)
        ]
    )


def hamming_dist(str1, str2):
    i = 0
    count = 0
    while i < len(str1):
        if str1[i] != str2[i]:
            count += 1
        i += 1
    return count


def hamming_dist_related_counts(
    num_spins: int, sprime_each_iter: list, states_accepted_each_iter: list
):

    dict_counts_states_hamming_dist = dict(
        zip(list(range(0, num_spins + 1)), [0] * (num_spins + 1))
    )
    ham_dist_s_and_sprime = np.array(
        [
            hamming_dist(states_accepted_each_iter[j], sprime_each_iter[j + 1])
            for j in range(0, len(states_accepted_each_iter) - 1)
        ]
    )
    for k in list(dict_counts_states_hamming_dist.keys()):
        dict_counts_states_hamming_dist[k] = np.count_nonzero(
            ham_dist_s_and_sprime == k
        )

    assert (
        sum(list(dict_counts_states_hamming_dist.values())) == len(sprime_each_iter) - 1
    )
    return dict_counts_states_hamming_dist


def energy_difference_related_counts(
    num_spins, sprime_each_iter: list, states_accepted_each_iter: list, model_in
):

    energy_diff_s_and_sprime = np.array(
        [
            abs(
                model_in.get_energy(sprime_each_iter[j])
                - model_in.get_energy(states_accepted_each_iter[j + 1])
            )
            for j in range(0, len(sprime_each_iter) - 1)
        ]
    )
    return energy_diff_s_and_sprime


# function to create dict for number of times states sprime were not accepted in MCMC iterations
def fn_numtimes_bitstring_not_accepted(list_after_trsn, list_after_accept, bitstring):

    where_sprime_is_bitstr = list(np.where(np.array(list_after_trsn) == bitstring)[0])
    where_bitstr_not_accepted = [
        k for k in where_sprime_is_bitstr if list_after_accept[k] != bitstring
    ]
    numtimes_sprime_is_bitstring = len(where_sprime_is_bitstr)
    numtimes_bitstring_not_accepted = len(where_bitstr_not_accepted)
    return numtimes_bitstring_not_accepted, numtimes_sprime_is_bitstring


def fn_states_not_accepted(
    list_states: list, list_after_trsn: list, list_after_accept: list
):
    list_numtimes_state_not_accepted = [
        fn_numtimes_bitstring_not_accepted(list_after_trsn, list_after_accept, k)[0]
        for k in list_states
    ]
    list_numtimes_sprime_is_state = [
        fn_numtimes_bitstring_not_accepted(list_after_trsn, list_after_accept, k)[1]
        for k in list_states
    ]
    dict_numtimes_states_not_accepted = dict(
        zip(list_states, list_numtimes_state_not_accepted)
    )
    dict_numtimes_sprime_is_state = dict(
        zip(list_states, list_numtimes_sprime_is_state)
    )
    return dict_numtimes_states_not_accepted, dict_numtimes_sprime_is_state


def int_to_str(state_obtained, nspin):
    return f"{state_obtained:0{nspin}b}"


def random_bstr(n, k=1):
    assert n >= k
    s = "1" * k + "0" * (n - k)
    s_list = list(s)
    random.shuffle(s_list)
    return "".join(s_list)


int_to_binary = lambda state_obtained, n_spins: f"{state_obtained:0{n_spins}b}"
binary_to_bipolar = lambda string: 2.0 * float(string) - 1.0


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

def observable_expectation(
    observable: callable, mcmc_chain: MCMCChain, skip_init: int = 100
):

    sample_observable = []
    for s in mcmc_chain.accepted_states:

        sample_observable.append(observable(s))

    sample_observable = np.array(sample_observable)

    return sample_observable.mean(dtype=float)  # , sample_observable.var(dtype= float)


###########################################################################################
## VISUALISATIONS AND PLOTTING ##
###########################################################################################
def plot_dict_of_running_avg_observable(
    dict_running_avg: dict, observable_legend_label: str
):
    plt.plot(
        list(dict_running_avg.keys()),
        list(dict_running_avg.values()),
        "-",
        label=observable_legend_label,
    )
    plt.xlabel("MCMC iterations")


def plot_bargraph_desc_order(
    desc_val_order_dict_in: dict,
    normalise_complete_data: bool = False,
    plot_first_few: int = -1,
    **bar_kwargs,
):
    width = 1.0
    list_keys = list(desc_val_order_dict_in.keys())
    list_vals = list(desc_val_order_dict_in.values())
    if normalise_complete_data:
        list_vals = np.divide(
            list_vals, sum(list_vals)
        )  # np.divide(list(vals), sum(vals))
    if plot_first_few != -1:
        plt.bar(list_keys[0:plot_first_few], list_vals[0:plot_first_few], **bar_kwargs)
    else:
        plt.bar(list_keys, list_vals, **bar_kwargs)
    plt.xticks(rotation=90)


def plot_multiple_bargraphs(
    list_of_dicts: list,
    list_labels: list,
    list_normalise: list,
    plot_first_few,
    sort_desc=False,
    sort_asc=False,
    figsize=(15, 7),
):
    list_keys = list(list_of_dicts[0].keys())
    dict_data = {}
    for i in range(0, len(list_labels)):
        # list_vals=[list_of_dicts[i][j] for j in list_keys if j in list(list_of_dicts[i].keys()) else 0] #list(list_of_dicts[i].values())
        list_vals = [
            list_of_dicts[i][j] if j in list(list_of_dicts[i].keys()) else 0
            for j in list_keys
        ]
        if list_normalise[i]:
            list_vals = np.divide(list_vals, sum(list_vals))
        dict_data[list_labels[i]] = list_vals
    df = pd.DataFrame(dict_data, index=list_keys)
    if sort_desc:
        df_sorted_desc = df.sort_values(list_labels[0], ascending=False)
        df_sorted_desc[:plot_first_few].plot.bar(rot=90, figsize=figsize)
    elif sort_asc:
        df_sorted_asc = df.sort_values(list_labels[0], ascending=True)
        df_sorted_asc[:plot_first_few].plot.bar(rot=90, figsize=figsize)
    elif sort_desc == False and sort_asc == False:
        df[:plot_first_few].plot.bar(rot=90, figsize=figsize)


def plot_hamming_distance_statistics(
    trajectory_stat_list: list, nspin: int, labels: list, figsize=(16, 8)
):

    plt.figure(figsize=figsize)

    bins = np.arange(0, nspin + 1)
    alpha = 0.3
    for item in zip(trajectory_stat_list, labels):

        alpha += (0.7) / len(trajectory_stat_list)
        plt.bar(
            *np.unique(item[0]["hamming"], return_counts=True),
            label=item[1],
            alpha=alpha,
        )

    # plt.xscale("log")

    plt.xlabel("Hamming-Distance Statistics")
    # plt.ylabel("Hamming Distance")
    plt.legend()
    plt.show()


def plot_acceptance_prob_statistics(
    trajectory_stat_list: list, labels: list, figsize=(15, 7)
):

    plt.figure(figsize=figsize)

    lcomp = []
    for tl in trajectory_stat_list:
        lcomp.append(np.min(tl["acceptance_prob"]))

    bins = np.linspace(np.log10(np.min(lcomp)) - 0.1, 0, num=30)

    alpha = 0.3

    for item in zip(trajectory_stat_list, labels):

        alpha += (0.7) / len(trajectory_stat_list)
        plt.hist(
            np.log10(item[0]["acceptance_prob"]),
            label=item[1],
            alpha=alpha,
            bins=bins,
            density=True,
        )
    plt.xlabel("Acceptance Probabilities | scale: log10")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.show()


# this function would be useful for plotting curves
def plot_with_error_band(
    xval: list,
    y_list_of_list: list,
    label: str,
    std_dev_multiplicative_factor: int = 0.5,
    alpha_for_plot: float = 0.5,
):
    # for i, data in dict_mcmc_bas_gridsize_3.items():
    #     print(i)
    #     keys = list(data.keys())
    #     for type in keys:
    #         print(type)
    #         print(name_replacement[type])
    #         data[name_replacement[type]] = data.pop(type)

    # name_replacement = {'cl': 'cl-uniform', 'local': 'cl-local-wt1', 'Q-MCMC:pauli_wt_1': 'qu-wt1', 'Q-MCMC:pauli_wt_3': 'qu-wt3', 'local_wt_3': 'cl-local-wt3'}
    curve_of_mean_value = np.mean(y_list_of_list, axis=0)
    standard_dev_band = np.std(y_list_of_list, axis=0)
    plt.plot(xval, curve_of_mean_value, "-", label=label)
    plt.fill_between(
        xval,
        curve_of_mean_value - standard_dev_band * std_dev_multiplicative_factor,
        curve_of_mean_value + standard_dev_band * std_dev_multiplicative_factor,
        alpha=alpha_for_plot,
    )


def plot_mcmc_iterations(DATA):
    mcmc_types = list(DATA.keys())
    iter_begin = 100
    iter_end = -1
    dim1 = int(len(mcmc_types) / 2) + 1
    dim2 = 2

    plt.figure(figsize=(40, 25))
    for j in range(1, len(mcmc_types) + 1):
        # print(j)
        plt.subplot(dim1, dim2, j)
        plt.plot(DATA[mcmc_types[j - 1]].markov_chain[iter_begin:iter_end])
        plt.title(mcmc_types[j - 1])

        # plt.legend()
    plt.show()


###########################################################################################
## BAS Datasets ##
###########################################################################################

from itertools import permutations, product


class bas_dataset:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        all_combn = ["".join(p) for p in product("01", repeat=self.grid_size)]
        all_combn.sort(key=lambda s: s.count("1"))
        all_combn.pop(0)
        all_combn.pop(-1)
        self.__all_combn = all_combn
        self.bas_dict = self.bars_and_stripes_dataset()
        self.dataset = (
            self.bas_dict["stripes"]
            + self.bas_dict["bars"]
            + self.bas_dict.get("both", [])
        )

    def vertical_stripes(self):
        vert_stripes = [j * self.grid_size for j in self.__all_combn]
        return vert_stripes

    def horizontal_bars(self):
        hor_bars = []
        for l in self.__all_combn:
            st = ""
            for j in l:
                st = st + j * self.grid_size
            hor_bars.append(st)
        return hor_bars

    def bars_and_stripes_dataset(self):
        bas_dict = {
            "stripes": self.vertical_stripes(),
            "bars": self.horizontal_bars(),
            # "both": ["0"*self.grid_size*self.grid_size, "1"*self.grid_size*self.grid_size]
        }
        return bas_dict

    ### create matrix of bitstring: meant for plotting
    def bit_string_to_2d_matrix(self, bitstring, array_shape: int):
        len_bs = len(bitstring)
        list_bs_int = [eval(i) for i in list(bitstring)]
        arr_bs = np.reshape(list_bs_int, (array_shape, array_shape))
        return arr_bs

    ### plot pixels
    def draw_pixelplot(self, bitstring: str, array_shape: int):
        im_array = self.bit_string_to_2d_matrix(bitstring, array_shape)
        plt.title(f"pixel plot for bitstring: {bitstring}")
        pixel_plot = plt.imshow(im_array, cmap="Greens", interpolation="nearest")
        plt.colorbar(pixel_plot)
        plt.show()


def hebbing_learning(list_bas_state: list):
    size = len(list_bas_state[0])
    wts = 0
    for i in list_bas_state:
        arr = np.array([-1 if elem == "0" else 1 for elem in i])
        array = np.reshape(arr, (size, 1))
        array_t = np.transpose(array)
        wts += array @ array_t
    wts = wts - len(list_bas_state) * np.identity(size)
    return wts


def get_cardinality_dataset(n_qubits, card=2):
    def generate_binary_strings(bit_count):
        binary_strings = []

        def genbin(n, bs=""):
            if len(bs) == n:
                binary_strings.append(bs)
            else:
                genbin(n, bs + "0")
                genbin(n, bs + "1")

        genbin(bit_count)
        return binary_strings

    binary_strings = generate_binary_strings(n_qubits)
    return [b for b in binary_strings if b.count("1") == card]
