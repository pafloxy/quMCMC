###################
# from dataclasses import dataclass 
from .energy_models import IsingEnergyFunction
from typing import Optional
from abc import ABC, abstractmethod, abstractproperty

###################
# @dataclass
class mcmc_sampler(ABC) :

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

    

