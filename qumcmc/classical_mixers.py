from typing import Union, List

# from itertools import combinations
from abc import ABC, abstractmethod

import numpy as np

# from qulacs import QuantumCircuit

from .basic_utils import xor_strings, random_bstr, get_random_state


class ClassicalMixer(ABC):

    def __init__(self, num_spins: int) -> None:
        self.num_spins = num_spins
        # self.current_state = current_state
        self._precompute_properties()

    @abstractmethod
    def propose_transition(self): ...

    def _precompute_properties(self):
        pass


class UniformProposals(ClassicalMixer):

    def _init_(
        self,
        num_spins: int,
    ) -> None:
        super().__init__(num_spins)

    def __repr__(self):
        return f"UniformProposals({self.num_spins})"

    def propose_transition(self, current_state=None):
        assert len(current_state) == self.num_spins, "Wrong 'current_state' "
        return get_random_state(self.num_spins)


class FixedWeightProposals(ClassicalMixer):

    def __init__(self, num_spins: int, bodyness: int) -> None:
        self.bodyness = bodyness
        assert self.bodyness <= num_spins, "Incorrect"

        super().__init__(num_spins)

    def __repr__(self):
        return f"FixedWeightProposals({self.num_spins}, ({self.bodyness}))"

    def propose_transition(self, current_state: str):
        assert len(current_state) == self.num_spins, "Wrong 'current_state' "
        rbstr = random_bstr(self.num_spins, self.bodyness)
        return xor_strings(current_state, rbstr)


class CustomProposals(ClassicalMixer):

    def __init__(self, num_spins: int, flip_pattern: List[int]) -> None:
        self.flip_pattern = flip_pattern
        super().__init__(num_spins)

    def __repr__(self):
        return f"CustomProposals({self.num_spins}, pattern: {self.flip_pattern})"

    def propose_transition(self, current_state: str):
        assert len(current_state) == self.num_spins, "Wrong 'current_state' "
        rbstr = ""
        __ = ["1" if i in self.flip_pattern else "0" for i in range(self.num_spins)]
        for _ in __:
            rbstr += _
        return xor_strings(current_state, rbstr)


class CombineProposals(ClassicalMixer):

    def __init__(
        self, proposal_methods: List[ClassicalMixer], probabilities: List[float]
    ) -> None:
        # super().__init__(num_spins)
        self.num_spins = proposal_methods[0].num_spins
        assert all(
            self.num_spins == pm.num_spins for pm in proposal_methods
        ), "Mixers don't have the same number of qubits"
        assert len(probabilities) == len(
            proposal_methods
        ), "Length of list of mixers and probabilities is not equal"

        self.probabilities = np.array(probabilities) / sum(probabilities)
        self.proposal_methods = proposal_methods

    def __repr__(self):
        return f"CombinedProposals( num-proposal-methods = {len(self.proposal_methods)} , p = {self.probabilities})"

    def propose_transition(self, current_state: str):
        assert len(current_state) == self.num_spins, "Wrong 'current_state' "
        proposal_method = np.random.choice(self.proposal_methods, p=self.probabilities)
        return proposal_method.propose_transition(current_state)
