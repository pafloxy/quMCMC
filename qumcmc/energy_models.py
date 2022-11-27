###########################################################################################
## IMPORTS ##
###########################################################################################

from basic_utils import *
from typing import Dict

###########################################################################################
## ENERGY MODEL ##
###########################################################################################

class IsingEnergyFunction:
    """ A class to build the Ising Energy Function from data  
    """

    def __init__(self, J: np.array, h: np.array, beta: float = 1.0) -> None:
        """
            ARGS:
            ----
            J: weight-matrix of the interactions between the spins 
            h: local field to the spins 

        """
        self.J = J
        self.h = h
        self.beta = beta
        self.num_spins = len(h)
    
    @property
    def get_J(self):
        return self.J
    
    @property
    def get_h(self):
        return self.h

    def get_energy(self, state: Union[str, np.array]) -> float:
        """ Returns the energy of a given state

            ARGS:
            ----
            state : configuration of spins for which the energy is requrieed to be calculated.
                    NOTE:  if input is an numpy array then it should only consist of bipolar values -> {+1, -1}

        """

        if isinstance(state, str):
            state = np.array([-1 if elem == "0" else 1 for elem in state])
            # state = np.array( [int(list(state)[i]) for i in range(len(state))])
            energy = 0.5 * np.dot(state.transpose(), self.J.dot(state)) + np.dot(
                self.h.transpose(), state
            )
            return energy
        else:
            return 0.5 * np.dot(state.transpose(), self.J.dot(state)) + np.dot(
                self.h.transpose(), state
            )


    def get_boltzmann_prob(
        self, state: Union[str, np.array], beta: float = 1.0
    ) -> float:

        """ Get un-normalised boltzmann probability of a given state 

            ARGS:
            ----
            state : configuration of spins for which probability is to be calculated 
            beta : inverse temperature (1/T) at which the probability is to be calculated.
        
        """
        return np.exp(-1 * beta * self.get_energy(state))

    def get_boltzmann_distribution(
        self, beta:float = 1.0, sorted:bool = False, plot_dist:bool = False
    ) -> dict :
        """ Get normalised boltzmann distribution over states 

            ARGS:
            ----
            beta : inverse temperature (1/ T)
            sorted  : if True then the states are sorted in in descending order of their probability
            plot_dist : if True then plots histogram corresponding to the boltzmann distribution

            RETURNS:
            -------
            'dict' corresponding to the distribution
        """
        all_configs = [f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))]
        bltzmann_probs = dict( [ ( state, self.get_boltzmann_prob(state, beta= beta) ) for state in all_configs ] )
        partition_sum=np.sum(np.array(list(bltzmann_probs.values())))
        prob_vals=list(np.array(list(bltzmann_probs.values()))*(1./partition_sum))

        bpd= dict(zip(all_configs, prob_vals ))
        bpd_sorted_desc= value_sorted_dict( bpd, reverse=True )
        
        if plot_dist:
                plt.figure(2)
                plot_bargraph_desc_order(bpd_sorted_desc, label="analytical",plot_first_few=30); plt.legend()
        
        
        if sorted: 
            return bpd_sorted_desc
        else :
            return bpd



    def get_observable_expectation(self, observable, beta: float = 1.0) -> float:
        """ Return expectation value of a classical observables

            ARGS :
            ----
            observable: Must be a function of the spin configuration which takes an 'np.array' of binary elements as input argument and returns a 'float'
            beta: inverse temperature

        """
        all_configs = np.array(list(itertools.product([1, 0], repeat=self.num_spins)))
        partition_sum = sum([self.get_boltzmann_prob(config) for config in all_configs])

        return sum(
            [
                self.get_boltzmann_prob(config)
                * observable(config)
                * (1 / partition_sum)
                for config in all_configs
            ]
        )

    def get_kldiv(self, q: dict, beta: float) -> float :
        """ Return calculated KL-divergence of the boltzmann distribution wrt. a given distribution

            ARGS:
            ----
            q : given distribution 
            beta : inverse temperature of the model 
        
        """
        ## check q 
        q_vals = list(q.values())
        bltz_dist = self.get_boltzmann_distribution(beta= beta, sorted= True)

        assert np.sum(q_vals) == 1 , " given distribution is not normalised "
        all_configs = [f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))]
        assert set(q.keys()).issubset(all_configs) , " given distribution is not defined over all possible configurations " 

        ## 

