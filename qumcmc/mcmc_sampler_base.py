###################
# from dataclasses import dataclass
from .energy_models import IsingEnergyFunction, Exact_Sampling
from .classical_mixers import ClassicalMixer
from .mixers import Mixer
from .classical_mcmc_routines import classical_mcmc
from .quantum_mcmc_routines import quantum_enhanced_mcmc
from .basic_utils import observable_expectation

from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod, abstractproperty

###################
class ExactSampler(Exact_Sampling):
    def __init__(self, model: IsingEnergyFunction, temperature: float = 1.0, verbose=False, name= 'EXACT') -> None:
        self.model = model
        self.num_spins = model.num_spins 
        self.temperature = temperature
        self.verbose = verbose
        self.name = name
    
    def __repr__(self):
        return super().__repr__()
    
    def run(self):
        super().__init__(self.model, 1/self.temperature, self.verbose)
        self.run_exact_sampling(1/self.temperature, verbose= self.verbose)


###################
# @dataclass
class MCMCSampler(ABC):

    def __init__(
        self,
        n_hops: int,
        model: IsingEnergyFunction,
        initial_state: Optional[str] = None,
        temperature: float = 1.0,
        verbose: bool = False,
        name: str = "MCMC",
    ) -> None:

        self.n_hops = n_hops
        self.model = model
        self.initial_state = initial_state
        self.temperature = temperature
        self.verbose = verbose
        self.name = name

    def __repr__(self):
        return f" MCMCSampler:{self.name}| model:{self.model}"

    @abstractmethod
    def run(self, save_mcmc:bool= True): ...

    @abstractmethod
    def get_observable_expectation(self , observable: callable, skip_init:int= 100): ...
        


## wrapper class for running classical_mcmc()
class ClassicalMCMCSampler(MCMCSampler):

    def __init__(
        self,
        n_hops: int,
        model: IsingEnergyFunction,
        proposition_method: ClassicalMixer,
        initial_state: Optional[str] = None,
        temperature: float = 1,
        verbose: bool = False,
        name: str = "Cl-MCMC",
    ) -> None:

        self.proposition_method = proposition_method
        super().__init__(n_hops, model, initial_state, temperature, verbose, name)

    def __repr__(self):
        return f"Classical MCMCSampler: {self.name} \n ------------------- Iterations : {self.n_hops} \n Model : {self.model} \n PropositionType : {self.proposition_method} \n initial-state : {self.initial_state} \ntemp : {self.temperature}"
        # pass

    def run(self, save_mcmc:bool= True):
        mcmc = classical_mcmc(
            n_hops=self.n_hops,
            model=self.model,
            proposition_method=self.proposition_method,
            initial_state=self.initial_state,
            temperature=self.temperature,
            verbose=self.verbose,
            name=self.name,
        )

        if save_mcmc:
            self.mcmc = mcmc
        return mcmc 
    
    def get_observable_expectation(self, observable: callable, skip_init:int= 100): 
        if self.mcmc : 
            return observable_expectation(observable, self.mcmc, skip_init)
        else :
            raise ValueError(f"No saved MCMC, set 'save_mcmc= True' in '{self}.run()'")

## wrapper class for running quantum_enhanced_mcmc()


class QuantumMCMCSampler(MCMCSampler):

    def __init__(
        self,
        n_hops: int,
        model: IsingEnergyFunction,
        mixer: Mixer,
        gamma: float,
        initial_state: Optional[str] = None,
        temperature: float = 1,
        delta_time=0.8,
        verbose: bool = False,
        name: str = "Q-MCMC",
    ) -> None:

        self.mixer = mixer
        self.gamma = gamma
        self.delta_time = delta_time
        super().__init__(n_hops, model, initial_state, temperature, verbose, name)

    # super.__init__(n_hops, model, initial_state, temperature, verbose, name)
    def __repr__(self):
        return f"Quantum MCMCSampler: {self.name} \n ------------------- Iterations : {self.n_hops} \n Model : {self.model} \n Mixer : {self.mixer} \n initial-state : {self.initial_state} \ngamma : {self.gamma} | temp : {self.temperature} | evolution-time : {self.delta_time}\n  "
        # pass

    def run(self, save_mcmc:bool= True):
        mcmc = quantum_enhanced_mcmc(
            n_hops=self.n_hops,
            model=self.model,
            mixer=self.mixer,
            gamma=self.gamma,
            initial_state=self.initial_state,
            temperature=self.temperature,
            delta_time=self.delta_time,
            verbose=self.verbose,
            name=self.name,
        )

        if save_mcmc:
            self.mcmc = mcmc
        return mcmc 

    def get_observable_expectation(self, observable: callable, skip_init:int= 100): 
        if self.mcmc : 
            return observable_expectation(observable, self.mcmc, skip_init)
        else :
            raise ValueError(f"No saved MCMC, set 'save_mcmc= True' in '{self}.run()'")
