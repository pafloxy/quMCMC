import numpy as np
from qulacs import QuantumCircuit
from typing import Union, List, Tuple, Optional
from itertools import combinations
from abc import ABC, abstractmethod


class Mixer(ABC):

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self._precompute_properties()

    @abstractmethod
    def get_mixer_circuit(self, delta_dime: float): ...

    def _precompute_properties(self):
        pass


class GenericMixer(Mixer):

    def __init__(
        self,
        n_qubits: int,
        bodyness: int,
    ) -> None:
        self.bodyness = bodyness
        super().__init__(n_qubits)

    def __repr__(self):
        return f" GenericMixer : \n ------------- \n num-qubits = {self.n_qubits } | bodyness = {self.bodyness}"

    def get_mixer_circuit(self, gamma: float, delta_time: float):

        qc_evolution_under_mixer = QuantumCircuit(self.n_qubits)

        for i in range(0, len(self.possible_qubit_combinations)):

            target_qubits_list = self.possible_qubit_combinations[i]
            pauli_id_single_term_in_mixer = [1] * len(target_qubits_list)
            angle = (
                -1 * 2 * gamma * delta_time
            )  # additional -1 factor since in qulacs convention is +1 in the rotation operator

            qc_evolution_under_mixer.add_multi_Pauli_rotation_gate(
                index_list=target_qubits_list,
                pauli_ids=pauli_id_single_term_in_mixer,
                angle=angle,
            )

        return qc_evolution_under_mixer

    def _precompute_properties(self):
        qubit_indices = range(self.n_qubits)
        self.possible_qubit_combinations = [
            list(i) for i in combinations(qubit_indices, self.bodyness)
        ]

    @classmethod
    def OneBodyMixer(cls, n_qubits: int):
        return cls(n_qubits, 1)

    @classmethod
    def TwoBodyMixer(cls, n_qubits: int):
        return cls(n_qubits, 2)

    @classmethod
    def ThreeBodyMixer(cls, n_qubits: int):
        return cls(n_qubits, 3)


class CustomMixer(Mixer):

    def __init__(
        self,
        n_qubits: int,
        possible_qubit_combinations: List[int],
    ) -> None:

        super().__init__(n_qubits)
        self.possible_qubit_combinations = possible_qubit_combinations

        self._precompute_properties()

    def __repr__(self):
        return f" GenericMixer : \n ------------- \n num-qubits = {self.n_qubits } \n -------- \n possible-qubit-combintations = {self.possible_qubit_combinations}"

    def get_mixer_circuit(self, gamma, delta_time):

        qc_evolution_under_mixer = QuantumCircuit(self.n_qubits)
        for i in range(0, len(self.possible_qubit_combinations)):

            target_qubits_list = self.possible_qubit_combinations[i]
            pauli_id_single_term_in_mixer = [1] * len(target_qubits_list)
            angle = (
                -1 * 2 * gamma * delta_time
            )  # additional -1 factor since in qulacs convention is +1 in the rotation operator

            qc_evolution_under_mixer.add_multi_Pauli_rotation_gate(
                index_list=target_qubits_list,
                pauli_ids=pauli_id_single_term_in_mixer,
                angle=angle,
            )
        return qc_evolution_under_mixer


class CoherentMixerSum(Mixer):

    def __init__(self, mixers: List[Mixer], weights: List[float]) -> None:
        self.n_qubits = mixers[0].n_qubits
        assert all(
            self.n_qubits == mixer.n_qubits for mixer in mixers
        ), "Mixers don't have the same number of qubits"
        assert len(weights) == len(
            mixers
        ), "Length of list of mixers and weights is not equal"
        self.mixers = mixers
        self.weights = np.array(weights) / sum(weights)

    def __repr__(self):
        return f"CoherentMixerSum -> num_mixers : {len(self.mixers)} |  weights : {self.weights}"

    def get_mixer_circuit(self, gamma: float, delta_time: float):
        total_circuit = QuantumCircuit(self.n_qubits)
        for weight, mixer in zip(self.weights, self.mixers):
            total_circuit.merge_circuit(
                mixer.get_mixer_circuit(gamma * weight, delta_time)
            )
        return total_circuit


class IncoherentMixerSum(Mixer):

    def __init__(self, mixers: List[Mixer], probabilities: List[float]) -> None:
        self.n_qubits = mixers[0].n_qubits
        assert all(
            self.n_qubits == mixer.n_qubits for mixer in mixers
        ), "Mixers don't have the same number of qubits"
        assert len(probabilities) == len(
            mixers
        ), "Length of list of mixers and probabilities is not equal"
        self.mixers = mixers
        self.probabilities = np.array(probabilities) / sum(probabilities)

    def __repr__(self):
        return f"IncoherentMixerSum -> num_mixers : {len(self.mixers)} |  probabilities : {self.probabilities}"

    def get_mixer_circuit(self, gamma: float, delta_time: float):
        mixer = np.random.choice(self.mixers, p=self.probabilities)
        return mixer.get_mixer_circuit(gamma, delta_time)
