###########################################################################################
## IMPORTS ##
###########################################################################################

## basic imports ##
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image
from numpy import pi
from tqdm import tqdm, trange, tgrange
from numpy import log2

import itertools
import math
from collections import Counter
from typing import Iterable, Mapping, Optional, Union
from collections import OrderedDict

import matplotlib.pyplot as plt

## qiskit imports ##
from qiskit import *
from qiskit import (  # , InstructionSet
    IBMQ,
    Aer,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    quantum_info,
)
from qiskit.algorithms import *
from qiskit.circuit.library import *
from qiskit.circuit.library import RXGate, RZGate, RZZGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.extensions import HamiltonianGate, UnitaryGate
from qiskit.quantum_info import Pauli, Statevector, partial_trace, state_fidelity
from qiskit.utils import QuantumInstance
from qiskit.visualization import (
    plot_bloch_multivector,
    plot_bloch_vector,
    plot_histogram,
    plot_state_qsphere,
)

## qiskit-backends ##
qsm = Aer.get_backend("qasm_simulator")
stv = Aer.get_backend("statevector_simulator")
aer = Aer.get_backend("aer_simulator")

###########################################################################################
## HELPER FUNCTIONS ##
###########################################################################################

def uncommon_els_2_lists(list_1,list_2):
  return list(set(list_1).symmetric_difference(set(list_2)))

def merge_2_dict(dict1, dict2):
    return({**dict1,**dict2})

def sort_dict_by_keys(dict_in:dict):
  
  return dict(OrderedDict(sorted(dict_in.items())))


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
    label: str,
    normalise_complete_data: bool = False,
    plot_first_few: int = 0,
):
    width = 1.0
    list_keys = list(desc_val_order_dict_in.keys())
    list_vals = list(desc_val_order_dict_in.values())
    if normalise_complete_data:
        list_vals = np.divide(
            list_vals, sum(list_vals)
        )  # np.divide(list(vals), sum(vals))
    if plot_first_few != 0:
        plt.bar(list_keys[0:plot_first_few], list_vals[0:plot_first_few], label=label)
    else:
        plt.bar(list_keys, list_vals, label=label)
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
