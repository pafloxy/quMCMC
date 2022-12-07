from .prob_dist import *


class trajectory_processing:
    '''  
    A class to use list_of_samples for different calculations 
    
    Method given here (but not limited to) can
    be come in handy for calculations of interest.
    '''
    def __init__(self, list_of_samples_sampled:list):
        self.list_samples=list_of_samples_sampled
        self.num_mcmc_steps= len(list_of_samples_sampled)
        self.num_spins=len(list_of_samples_sampled[0])
        self.all_poss_samples=states(num_spins=len(list_of_samples_sampled[0]))
        # import states function from basic utils.py
        # from qumcmc.prob_dist import *
        self.dict_count=self.count_states_occurence(list_samples= self.list_samples)
        self.dict_distn=self.empirical_distn()# not efficient! that is why I dont want inplace replacement
    
    def count_states_occurence(self,list_samples)->DiscreteProbabilityDistribution:
        ''' 
        Function to get dict of occurence count of sample

        Returns: instance of DiscreteProbabilityDistrubution
        '''
        dict_count=DiscreteProbabilityDistribution(
            dict(zip(self.all_poss_samples,[0]*(len(self.all_poss_samples))))
        )
        dict_count.update(dict(Counter(list_samples)))
        return dict_count
    
    def empirical_distn(self)->DiscreteProbabilityDistribution:
        ''' 
        Function to get dict of empirical distn from list of samples M.Chain was in.

        Returns: an instance of DiscreteProbabilityDistrubution
        '''
        dict_distn=DiscreteProbabilityDistribution(
            dict(zip(self.all_poss_samples,[0]*(len(self.all_poss_samples))))
        )
        update_with=DiscreteProbabilityDistribution(dict(Counter(self.list_samples)))
        update_with._normalise()
        dict_distn.update(update_with)
        return dict_distn
    
    def running_avg_magnetization_as_list(self)->np.array:
        """
        Function to calculate the running average magnetization for the given mcmc trajectory as list
        
        Args:
        list_states_mcmc= List of state markov chain is in after each MCMC step
        
        Returns: numpy array of running value of magnetization

        """
        list_of_strings = self.list_samples
        list_of_lists = (
            np.array([list(int(s) for s in bitstring) for bitstring in list_of_strings]) * 2
            - 1
        )
        return np.array(
            [
                np.mean(np.sum(list_of_lists, axis=1)[:ii])
                for ii in range(1, len(self.list_samples) + 1)
            ]
        )

    def average_of_some_observable(self,dict_observable_val_at_states: dict):
        return avg(dict_probabilities=self.dict_distn, dict_observable_val_at_states=dict_observable_val_at_states)