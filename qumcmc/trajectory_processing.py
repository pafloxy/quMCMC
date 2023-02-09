from .prob_dist import *
from .energy_models import IsingEnergyFunction, Exact_Sampling

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
    
    def running_js_divergence(self, actual_boltz_distn:DiscreteProbabilityDistribution):
       
        list_chain_state_accepted = self.list_samples
        num_nhops=len(list_chain_state_accepted)
        list_js_after_each_step=[]
        possible_states=list(actual_boltz_distn.keys())
        time_sec1=[];time_sec2=[]
        num_spins=len(list_chain_state_accepted[0])
        poss_states=states(num_spins=num_spins) 
        for step_num in tqdm(range(100,num_nhops, 100)): ##pafloxy : starting at 100 instead of 0 , neglecting intital states m.chain was in

            temp_distn_model=dict(zip(possible_states,[0]*(len(possible_states))))

            temp_distn_model.update(get_distn(list_chain_state_accepted[50:step_num]))  ##pafloxy : starting from 50, neglecting inital state m.chain was in 

            js_temp=js_divergence(actual_boltz_distn,temp_distn_model)

            list_js_after_each_step.append(js_temp)


        return list_js_after_each_step

##############################################################################################
## Functions to process trajectory data from MCMChain instances ##
##############################################################################################

def calculate_running_kl_divergence(actual_boltz_distn: DiscreteProbabilityDistribution, mcmc_chain: MCMCChain, skip_steps: int = 1, verbose= False) -> list:
    num_nhops = len(mcmc_chain.states)
    
    list_kl_after_each_step=[]

    for step_num in tqdm(range(1, num_nhops, skip_steps), disable= not verbose): ##pafloxy : starting at 100 instead of 0 , neglecting effect of intital states

        temp_distn_model = mcmc_chain.get_accepted_dict(normalize=True, until_index=step_num)

        kl_temp=kl_divergence(actual_boltz_distn,temp_distn_model)

        list_kl_after_each_step.append(kl_temp)


    return list_kl_after_each_step

def calculate_running_js_divergence(actual_boltz_distn: DiscreteProbabilityDistribution, mcmc_chain: MCMCChain, skip_steps: int = 1) -> list:
    num_nhops = len(mcmc_chain.states)
    
    list_js_after_each_step=[]

    for step_num in tqdm(range(1, num_nhops, skip_steps)): ##pafloxy : starting at 100 instead of 0 , neglecting effect of intital states

        temp_distn_model = mcmc_chain.get_accepted_dict(normalize=True, until_index=step_num)

        js_temp=js_divergence(actual_boltz_distn,temp_distn_model, prelim_check= False)

        list_js_after_each_step.append(js_temp)


    return list_js_after_each_step

def calculate_runnning_magnetisation(mcmc_chain: MCMCChain, skip_steps: int = 1) -> list:    
    
    num_nhops = len(mcmc_chain.states)

    list_mag_after_each_step=[]    

    magnetisation_dict = dict([ (state, magnetization_of_state(state) ) for state  in  mcmc_chain.accepted_states ])

    for step_num in tqdm(range(1, num_nhops, skip_steps)): ##pafloxy : starting at 100 instead of 0 , neglecting effect of intital states

        temp_distn_model = DiscreteProbabilityDistribution( mcmc_chain.get_accepted_dict(normalize=True, until_index=step_num) )
        
        mag_temp = temp_distn_model.expectation(magnetisation_dict)

        list_mag_after_each_step.append(mag_temp)
    
    return list_mag_after_each_step

from typing import Union        
def get_trajectory_statistics(mcmc_chain: MCMCChain, model: Union[IsingEnergyFunction, Exact_Sampling],to_observe:set = {'acceptance_prob','kldiv', 'hamming', 'energy', 'magnetisation'} ,verbose:bool= False):

    trajectory = mcmc_chain.states

    acceptance_prob = lambda si, sf: min(1, model.get_boltzmann_factor(sf.bitstring) / model.get_boltzmann_factor(si.bitstring) )
    hamming_diff = lambda si, sf: hamming_dist(si.bitstring, sf.bitstring)
    energy_diff = lambda si, sf: model.get_energy(sf.bitstring) - model.get_energy(si.bitstring)

    acceptance_statistic = [];hamming_statistic = [];energy_statistic = []
    
    current_state_index = 0; proposed_state_index = current_state_index + 1

    while proposed_state_index < len(trajectory) :

        
        if verbose : print('trans: '+ str(trajectory[current_state_index].bitstring) + ' -> '+ str(trajectory[proposed_state_index].bitstring)+" status: "+str(trajectory[proposed_state_index].accepted)  )

        if 'acceptance_prob' in to_observe: acceptance_statistic.append( acceptance_prob(trajectory[current_state_index], trajectory[proposed_state_index] ) )
        if 'energy' in to_observe: energy_statistic.append( energy_diff(trajectory[current_state_index], trajectory[proposed_state_index] ) )
        if 'hamming' in to_observe: hamming_statistic.append( hamming_diff(trajectory[current_state_index], trajectory[proposed_state_index] ) )

        
        if trajectory[proposed_state_index].accepted :
            current_state_index =  proposed_state_index
        
        proposed_state_index += 1

    trajectory_statistics = {}
    if 'acceptance_prob' in to_observe: trajectory_statistics['acceptance_prob'] = np.array(acceptance_statistic)
    if 'energy' in to_observe: trajectory_statistics['energy'] = np.array(energy_statistic)
    if 'hamming' in to_observe: trajectory_statistics['hamming'] = np.array(hamming_statistic)
    if 'kldiv' in to_observe:
        rkl = calculate_running_kl_divergence(model.boltzmann_pd, mcmc_chain, skip_steps= 1)
        trajectory_statistics['kldiv'] = np.array(rkl)

    return trajectory_statistics