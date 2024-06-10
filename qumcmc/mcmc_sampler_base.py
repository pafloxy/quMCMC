###################
# from dataclasses import dataclass 
from .energy_models import IsingEnergyFunction
from .classical_mixers import ClassicalMixer
from .mixers import Mixer 
from .classical_mcmc_routines import classical_mcmc
from .quantum_mcmc_routines import quantum_enhanced_mcmc 

from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod, abstractproperty
###################
# @dataclass
class MCMCSampler(ABC) :

    def __init__(self,
        n_hops: int ,
        model : IsingEnergyFunction, 
        initial_state : Optional[str] = None,
        temperature : float  = 1.0 ,
        verbose: bool = False ,
        name : str = "MCMC" ) -> None : 

        self.n_hops= n_hops 
        self.model= model 
        self.initial_state = initial_state 
        self.temperature = temperature
        self.verbose = verbose
        self.name = name 
        
    def __repr__(self): 
        return f" MCMCSampler:{self.name}| model:{self.model}"
    
    @abstractmethod
    def run(self): ... 

    

## wrapper class for running classical_mcmc() 
class ClassicalMCMCSampler(MCMCSampler) : 
    
    def __init__(self, 
        n_hops: int ,
        model: IsingEnergyFunction,
        proposition_method: ClassicalMixer,
        initial_state: Optional[str] = None,
        temperature: float = 1,
        verbose: bool = False,
        name: str = "Cl-MCMC") -> None :  
        
        self.proposition_method = proposition_method
        super().__init__(n_hops, model, initial_state, temperature, verbose, name)

    def __repr__(self):
        return f"Classical MCMCSampler: {self.name} \n ------------------- Iterations : {self.n_hops} \n Model : {self.model} \n PropositionType : {self.proposition_method} \n initial-state : {self.initial_state} \ntemp : {self.temperature}" 
        # pass 
    def run(self): 
        return classical_mcmc(n_hops= self.n_hops ,
                                     model= self.model , 
                                     proposition_method= self.proposition_method ,
                                     initial_state= self.initial_state , 
                                     temperature= self.temperature ,                                      
                                     verbose= self.verbose , 
                                     name = self.name)

## wrapper class for running quantum_enhanced_mcmc() 

class QuantumMCMCSampler(MCMCSampler) : 

    def __init__(self,
        n_hops: int ,
        model: IsingEnergyFunction,
        mixer: Mixer,
        gamma: GammaType,
        initial_state: Optional[str] = None,
        temperature: float = 1,
        delta_time=0.8,
        verbose: bool = False,
        name: str = "Q-MCMC") -> None : 
        
        self.mixer = mixer 
        self.gamma = gamma 
        self.delta_time = delta_time
        super().__init__(n_hops, model, initial_state, temperature, verbose, name)

    # super.__init__(n_hops, model, initial_state, temperature, verbose, name)
    def __repr__(self):
        return f"Quantum MCMCSampler: {self.name} \n ------------------- Iterations : {self.n_hops} \n Model : {self.model} \n Mixer : {self.mixer} \n initial-state : {self.initial_state} \ngamma : {self.gamma} | temp : {self.temperature} | evolution-time : {self.delta_time}\n  " 
        # pass 
    def run(self): 
        return quantum_enhanced_mcmc(n_hops= self.n_hops ,
                                     model= self.model , 
                                     mixer = self.mixer , 
                                     gamma= self.gamma ,
                                     initial_state= self.initial_state , 
                                     temperature= self.temperature , 
                                     delta_time= self.delta_time , 
                                     verbose= self.verbose , 
                                     name = self.name)
