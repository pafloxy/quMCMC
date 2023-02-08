###########################################################################################
## IMPORTS ##
###########################################################################################

from .basic_utils import *
from typing import Dict
from copy import deepcopy
import pandas as pd
from typing import Union
import seaborn as sns
import random

# import  
from .basic_utils import *
from .energy_models import IsingEnergyFunction, Exact_Sampling
from .classical_mcmc_routines import *
from .quantum_mcmc_routines_qulacs import *     #for qulacs Simulator backend
# from .quantum_mcmc_routines_qulacs import quantum_enhanced_mcmc   #for qiskit Aer's Simulator backend 
from .trajectory_processing import *

from qulacs import QuantumState
from qulacs_core import DensityMatrix

from qiskit.visualization import plot_histogram
###########################################################################################
## HELPER FUNCTIONS ##
###########################################################################################

int_to_binary = lambda state_obtained, n_spins : f"{state_obtained:0{n_spins}b}"
binary_to_bipolar = lambda string : 2.0 * float(string) - 1.0

def get_observable_expectation(observable: callable, mcmc_chain: MCMCChain, skip_init:int= 100) :

    sample_observable = []
    for s in mcmc_chain.accepted_states :

        sample_observable.append(observable(s) )
    
    sample_observable = np.array(sample_observable)
        
    return sample_observable.mean(dtype= float)#, sample_observable.var(dtype= float)


def correlation_spins(state: str, indices : Union[tuple, List] ):

    assert len(indices) <= len(state)
    
    
    prod = 1.0
    for j in indices :
        prod *= binary_to_bipolar(state[j])

    return prod

def cd_J(index, data_distribution:DiscreteProbabilityDistribution, mcmc_chain: MCMCChain):

    assert len(index) == 2
    observable = lambda s: correlation_spins(s, [index[0], index[1]])
    r = data_distribution.get_observable_expectation(observable) - get_observable_expectation(observable, mcmc_chain, skip_init= 100)

    return r

def cd_h(index:int, data_distribution:DiscreteProbabilityDistribution, mcmc_chain: MCMCChain):

    assert isinstance(index, int)
    observable = lambda s: correlation_spins(s, [index])
    r = data_distribution.get_observable_expectation(observable) - get_observable_expectation(observable, mcmc_chain, skip_init= 100)

    return r

# @dataclass
class cd_training():
    ''' 
    model: initial model = (J init, h init) at some temp T
    beta: 1/Temperature
    data_dist: empirical data which we want to learn!
    
    '''

    def __init__(self, model: IsingEnergyFunction, beta:float ,data_dist: DiscreteProbabilityDistribution) -> None:
        self.model = deepcopy(model)
        self.model_beta = beta
        self.data_distribution = data_dist
        self.training_history = {}
        self.kl_div = []
        self.list_pair_of_indices=[[i,j] for i in range(1,self.model.num_spins) for j in range(i,self.model.num_spins) if j!=i]
        

    def cd_J(self, index, mcmc_chain: MCMCChain):

        assert len(index) == 2
        observable = lambda s: correlation_spins(s, [index[0], index[1]])
        r = self.data_distribution.get_observable_expectation(observable) - get_observable_expectation(observable, mcmc_chain, skip_init= 100)

        return r

    def cd_h(self, index:int, mcmc_chain: MCMCChain):

        assert isinstance(index, int)
        observable = lambda s: correlation_spins(s, [index])
        r = self.data_distribution.get_observable_expectation(observable) - get_observable_expectation(observable, mcmc_chain, skip_init= 100)

        return r

    # @setattr
    # def data_distribution()
    
    def _train_on_mcmc_chain(self, lr:float= 0.01, 
    method = 'quantum-enhanced', 
    iterations: int = 100, # rename this iterations to something else
    num_random_Jij: int=10,
    mcmc_steps:int =1000 ):# we will try to increase mcmc steps. 

        random.seed(random.random()) ## add seed to random ##TODO
        if method == 'quantum-enhanced' :
            self.mcmc_chain = quantum_enhanced_mcmc(
            n_hops=mcmc_steps,
            model=self.model,
            temperature=1/self.model_beta,
            verbose= False
            )
        elif method == 'classical-uniform' : 
            self.mcmc_chain = classical_mcmc(
            n_hops=mcmc_steps,
            model=self.model,
            temperature=1/self.model_beta,
            verbose= False
            )
        
        ## random update strategy ##
        ## just realised that even this is not a good thing! 
        assert iterations<=self.model.num_spins, f"iterations should be <= num_spins (which is= {self.model.num_spins}) "
        assert num_random_Jij<=len(self.list_pair_of_indices), f"num_random_Jij should be <=len(self.list_pair_of_indices) (which is= {len(self.list_pair_of_indices)})"
        
        list_random_indices=random.sample(range(0,self.model.num_spins), iterations)
        #list_pair_of_indices=[[i,j] for i in range(1,self.model.num_spins) for j in range(i,self.model.num_spins) if j!=i]
        #list_pair_of_different_indices=random.sample(self.list_pair_of_indices,k=num_random_Jij)

        list_pair_of_different_indices=[[list_random_indices[j],
                    random.choice(list(range(0,list_random_indices[j]))+list(range(list_random_indices[j]+1,self.model.num_spins)))] 
                    for j in range(0,iterations)]
        
        # ## Update J
        for k in range(len(list_pair_of_different_indices)):
            indices_J=list_pair_of_different_indices[k]
            updated_param_j=self.model.J[indices_J[0],indices_J[1]] - lr * self.cd_J(indices_J, self.mcmc_chain)
            self.model._update_J(updated_param_j, indices_J)

        for k in range(iterations):
            #indices_J=list_pair_of_different_indices[k]
            #updated_param_j = model.J[indices_J[0],indices_J[1]] - lr * self.cd_J(indices_J, self.mcmc_chain)

            # update h
            index_h=list_random_indices[k]
            updated_param_h=self.model.h[index_h] - lr*self.cd_h(index_h,self.mcmc_chain)

            #self.model._update_J(updated_param_j, indices_J)
            self.model._update_h(updated_param_h, index_h)
            

    def train(self, lr:float= 0.01, method = 'quantum-enhanced', 
    epochs:int = 10, iterations: int = 100, num_random_Jij:int=5,
    mcmc_steps:int = 500, show_kldiv:bool = True ):

        ## random update strategy ##
        # kl_div = []
        iterator = tqdm(range(epochs), desc= 'training epochs')
        iterator.set_postfix({'method': method})
        for epoch in iterator:

            self._train_on_mcmc_chain(lr= lr , 
            method = method, iterations= iterations, num_random_Jij=num_random_Jij,
            mcmc_steps= mcmc_steps )

            if show_kldiv:
                
                # self.kl_div.append(kl_divergence(  self.data_distribution, self.mcmc_chain.get_accepted_dict(normalize= True)  ))
                
                exact_sampled_model = Exact_Sampling(self.model, self.model_beta)
                self.kl_div.append(kl_divergence(  self.data_distribution, exact_sampled_model.boltzmann_pd  ))
                
                iterator.set_postfix( { 'method ': method, 'kl div ' : self.kl_div[-1] })
        
        ## update training data ##
        # self.kl_div += kl_div
        self.training_history['kl_div']= self.kl_div
