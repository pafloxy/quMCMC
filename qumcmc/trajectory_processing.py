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
    if skip_steps > 1:
        print("skip_steps currently not available and defaults to 1")
    
        # TODO
        num_nhops = len(mcmc_chain.states)
    
    list_kl_after_each_step=[]

    ## slow
    # for step_num in tqdm(range(1, num_nhops, skip_steps), disable= not verbose): ##pafloxy : starting at 100 instead of 0 , neglecting effect of intital states

    #     temp_distn_model = mcmc_chain.get_accepted_dict(normalize=True, until_index=step_num)
    #     # temp_distn_model = mcmc_chain.get_accepted_dict(normalize=True)

    #     kl_temp=kl_divergence(actual_boltz_distn,temp_distn_model)

    #     list_kl_after_each_step.append(kl_temp)

    ## faster
    tar_probs = np.array([v for k, v in sorted(actual_boltz_distn.items())])
    nspin = len(list(actual_boltz_distn.keys())[0])
    mod_probs = np.zeros(2**nspin)

    for ii, bitstr in tqdm(enumerate(mcmc_chain.markov_chain, start=1), disable= not verbose): 
        
        mod_probs[int(bitstr, 2)] += 1

        kl_temp = vectoried_KL(tar_probs, mod_probs/ii)

        list_kl_after_each_step.append(kl_temp)

    return list_kl_after_each_step


def calculate_running_js_divergence(actual_boltz_distn: DiscreteProbabilityDistribution, mcmc_chain: MCMCChain, skip_steps: int = 1,prelim_check=False) -> list:
    num_nhops = len(mcmc_chain.states)
    
    list_js_after_each_step=[]

    for step_num in tqdm(range(1, num_nhops, skip_steps)): ##pafloxy : starting at 100 instead of 0 , neglecting effect of intital states

        temp_distn_model = mcmc_chain.get_accepted_dict(normalize=True, until_index=step_num)

        js_temp=js_divergence(actual_boltz_distn,temp_distn_model, prelim_check= prelim_check)

        list_js_after_each_step.append(js_temp)


    return list_js_after_each_step

def calculate_runnning_magnetisation(mcmc_chain: MCMCChain, skip_steps: int = 1, verbose:bool= False) -> list:    
    
    num_nhops = len(mcmc_chain.states)

    list_mag_after_each_step=[]    

    magnetisation_dict = dict([ (state, magnetization_of_state(state) ) for state  in  mcmc_chain.accepted_states ])

    for step_num in tqdm(range(1, num_nhops, skip_steps), disable= not verbose): ##pafloxy : starting at 100 instead of 0 , neglecting effect of intital states

        temp_distn_model = DiscreteProbabilityDistribution( mcmc_chain.get_accepted_dict(normalize=True, until_index=step_num) )
        
        mag_temp = temp_distn_model.expectation(magnetisation_dict)

        list_mag_after_each_step.append(mag_temp)
    
    return list_mag_after_each_step

from typing import Union        
def get_trajectory_statistics(mcmc_chain: MCMCChain, model: Union[IsingEnergyFunction, Exact_Sampling],
                                to_observe:set = {'acceptance_prob','kldiv', 'hamming', 'energy', 'magnetisation'} 
                                ,verbose:bool= False):

    trajectory = mcmc_chain.states

    # acceptance_prob = lambda si, sf: min(1, model.get_boltzmann_factor(sf.bitstring) / model.get_boltzmann_factor(si.bitstring) )
    hamming_diff = lambda si, sf: hamming_dist(si.bitstring, sf.bitstring)
    energy_diff = lambda si, sf: model.get_energy(sf.bitstring) - model.get_energy(si.bitstring)

    acceptance_statistic = [];hamming_statistic = [];energy_statistic = []
    
    current_state_index = 0; proposed_state_index = current_state_index + 1

    while proposed_state_index < len(trajectory) :

        
        if verbose : print('trans: '+ str(trajectory[current_state_index].bitstring) + ' -> '+ str(trajectory[proposed_state_index].bitstring)+" status: "+str(trajectory[proposed_state_index].accepted)  )

        # if 'acceptance_prob' in to_observe: acceptance_statistic.append( acceptance_prob(trajectory[current_state_index], trajectory[proposed_state_index] ) )
        if 'energy' in to_observe: energy_statistic.append( energy_diff(trajectory[current_state_index], trajectory[proposed_state_index] ) )
        #if 'hamming' in to_observe: hamming_statistic.append( hamming_diff(trajectory[current_state_index], trajectory[proposed_state_index] ) )

        
        if trajectory[proposed_state_index].accepted :
            current_state_index =  proposed_state_index
        
        proposed_state_index += 1

    ### acceptance probability 
    if 'acceptance_prob' in to_observe:
        current_bitstring = trajectory[0].bitstring
        counter = 1
        for st in trajectory:
            if st.accepted and st.bitstring!=current_bitstring: # not, and, or.
                acceptance_statistic.append(1/counter)
                current_bitstring = st.bitstring
                counter = 1
            else:
                counter += 1
        acceptance_statistic.append(1/counter)

    ### hamming distance caln
    num_spins=model.num_spins
    hamming_dist_keys=list(range(0,num_spins+1))
    init_val_keys=dict(zip(['accepted', 'rejected', 'total'],[0,0,0]))
    list_init_val_keys=[init_val_keys.copy() for i in range(0,len(hamming_dist_keys))]
    dict_hamming_distance_statistics=dict(zip(hamming_dist_keys, list_init_val_keys))
    
    current_bs_idx=0
    for proposed_bs_idx in range(1,len(trajectory)):
        hamming_dist_val=hamming_diff(si=trajectory[current_bs_idx],
                                        sf=trajectory[proposed_bs_idx])
        if trajectory[proposed_bs_idx].accepted:
            dict_hamming_distance_statistics[hamming_dist_val]['accepted']+=1
            current_bs_idx=proposed_bs_idx
        else:
            dict_hamming_distance_statistics[hamming_dist_val]['rejected']+=1
    
    for i in range(0,num_spins+1):
        dict_hamming_distance_statistics[i]['total']=dict_hamming_distance_statistics[i]['accepted']+dict_hamming_distance_statistics[i]['rejected']

    trajectory_statistics = {}
    if 'acceptance_prob' in to_observe: trajectory_statistics['acceptance_prob'] = np.array(acceptance_statistic)
    if 'energy' in to_observe: trajectory_statistics['energy'] = np.array(energy_statistic)
    if 'hamming' in to_observe: trajectory_statistics['hamming'] = dict_hamming_distance_statistics#np.array(hamming_statistic)
    if 'kldiv' in to_observe:
        rkl = calculate_running_kl_divergence(model.boltzmann_pd, mcmc_chain, skip_steps= 1)
        trajectory_statistics['kldiv'] = np.array(rkl)

    return trajectory_statistics

##############################################################################################
## Class to process MCMCData from multiple experiments ##
##############################################################################################
import os
import pickle
class ProcessMCMCData():
    
    
    def __init__(self, data: dict, model: Exact_Sampling, savefile_path: str = None, name: str= 'SAMPLINGRESULT'):

        self.data = data
        self.model = model
        self.processed_data = {}
        self.savefile_path = savefile_path
        self.name = name 
        
        self.seeds = list(self.data.keys())
        self.mcmc_types = list(self.data[1].keys())
        
        # os.chdir(self.savefile_path)
        print('OUTPUT DIRECTORY: ', os.path.join( os.getcwd(), self.savefile_path) )
    
    def PROCESS_ALL(self,save_data= True):
        
        self.CALCULATE_KL_DIV(save_data= save_data  )
        self.CALCULATE_MAGNETISATION(save_data= save_data  )
        self.CALCULATE_MCMC_STATISTICS(save_data= save_data  )
        if save_data :
            os.chdir(self.savefile_path)                
            with open(self.name + '.pkl','wb') as f:
                    pickle.dump(self,f)
            os.chdir('../..')
    
    def UPDATE_DATA(self, new_data: dict, new_seeds, new_mcmc_types, save_data = True):

        if self.processed_data == {} :
            raise ValueError(' method ''UPDATE_DATA'' cannot be called on unprocessed data, call "PROCESS_ALL" intially ')

        self.data = new_data
        self.CALCULATE_KL_DIV(save_data= save_data, allow_reprocessing=True, seeds_new= new_seeds, mcmc_types_new= new_mcmc_types )
        self.CALCULATE_MAGNETISATION(save_data= save_data, allow_reprocessing=True, seeds_new= new_seeds, mcmc_types_new= new_mcmc_types )
        self.CALCULATE_MCMC_STATISTICS(save_data= save_data, allow_reprocessing=True, seeds_new= new_seeds, mcmc_types_new= new_mcmc_types )
        
        self.seeds = list(self.data.keys())
        self.mcmc_types = list(self.data[1].keys())
    
    # def PLOT_KL_DIV(self, save_plot = False, mcmc_types_to_plot = 'all'):
    
    #     ## plotting
    #     x=list(range(0,15000+1))
    #     plt.figure(figsize=(12,8))
    #     if mcmc_types_to_plot == 'all': 
    #         mcmc_types_to_plot = self.processed_data['KL-DIV'].keys()
    #     for mcmc_type in mcmc_types_to_plot:
    #         plot_with_error_band(x, self.processed_data['KL-DIV'][mcmc_type] ,label= mcmc_type)
    
    #     plt.xlabel("iterations ")
    #     plt.ylabel("KL divergence")
    #     plt.yscale('log')
    #     plt.legend()

    #     if save_plot:
    #             os.chdir(self.savefile_path)      
    #             figname = self.name + 'KL-DIV.pdf'
    #             plt.savefig(figname)
        
    #             os.chdir('../..')        
        
    #     plt.show()

    def CALCULATE_KL_DIV(self, save_data:bool= False, allow_reprocessing= False, seeds_new = None, mcmc_types_new= None  ):
        
        not_computed = False
        ## data processing    
        if 'KL-DIV' not in self.processed_data:
            not_computed = True
            seeds = self.seeds
            mcmc_types = self.mcmc_types
            self.processed_data['KL-DIV'] = { mcmc_type: [] for mcmc_type in self.mcmc_types }

        if allow_reprocessing:
            if seeds_new == None: seeds = self.seeds
            else: seeds = seeds_new
            
            if mcmc_types_new == None: mcmc_types = self.mcmc_types 
            else: 
                mcmc_types = mcmc_types_new
                for mcmc_type in mcmc_types:
                    self.processed_data['KL-DIV'][mcmc_type] = []

        if not_computed or allow_reprocessing:
            for seed in tqdm(seeds):
                for mcmc_type in mcmc_types:
          
                    kldiv = calculate_running_kl_divergence(self.model.boltzmann_pd, self.data[seed][mcmc_type])
                    self.processed_data['KL-DIV'][mcmc_type].append(kldiv)

            if save_data :
                os.chdir(self.savefile_path)                
                with open(self.name + '.pkl','wb') as f:
                        pickle.dump(self,f)
                os.chdir('../..')


    # def PLOT_MAGNETISATION(self, save_plot= False , mcmc_types_to_plot = 'all'):
    
    #     ## plotting
    #     fig, ax1 = plt.subplots(figsize=(12,8))
    #     left, bottom, width, height = [0.55, 0.2, 0.25, 0.25]

    #     x=list(range(0,15000))
    #     # mcmc_types = self.data[1].keys()
    #     if mcmc_types_to_plot == 'all': 
    #         mcmc_types_to_plot = self.processed_data['MAGNETISATION'].keys()
        
    #     for mcmc_type in mcmc_types_to_plot:
            
    #         mean_mag_local=np.mean(self.processed_data['MAGNETISATION'][mcmc_type],axis=0)
    #         std_local=np.std(self.processed_data['MAGNETISATION'][mcmc_type],axis=0)
    #         ax1.plot(x,mean_mag_local,label= mcmc_type)
    #         ax1.fill_between(x,mean_mag_local-std_local/2,mean_mag_local+std_local/2,alpha=0.45)


    #     ax1.axhline(self.model.get_observable_expectation(magnetization_of_state), label= 'actual',linestyle='--')
    #     ax1.legend()
    #     ax1.set_xlabel("Iterations")
    #     ax1.set_ylabel("Magnetisation")
        
    #     if save_plot:       
    #             os.chdir(self.savefile_path)         
    #             figname = self.name + 'MAGNETISATION.pdf'
    #             plt.savefig(figname)
    #             os.chdir('../..')        
            
    #     plt.show()

    def CALCULATE_MAGNETISATION(self, save_data:bool = False , allow_reprocessing= False, seeds_new = None, mcmc_types_new= None  ):
        
        not_computed = False
        ## data processing
        if 'MAGNETISATION' not in self.processed_data:
            not_computed = True
            seeds = self.seeds
            mcmc_types = self.mcmc_types
            self.processed_data['MAGNETISATION'] = { mcmc_type: [] for mcmc_type in mcmc_types }
            magnetization_model = self.model.get_observable_expectation(magnetization_of_state)

        if allow_reprocessing:
            if seeds_new == None: seeds = self.seeds
            else: seeds = seeds_new
            
            if mcmc_types_new == None: mcmc_types = self.mcmc_types 
            else: 
                mcmc_types = mcmc_types_new
                for mcmc_type in mcmc_types:
                    self.processed_data['MAGNETISATION'][mcmc_type] = []


        if not_computed or allow_reprocessing:
        
            for seed in tqdm(seeds):
                for mcmc_type in mcmc_types:
                    mag = calculate_runnning_magnetisation(self.data[seed][mcmc_type], verbose= False)
                    self.processed_data['MAGNETISATION'][mcmc_type].append(mag)


            if save_data :
                os.chdir(self.savefile_path)
                
                with open(self.name + '.pkl','wb') as f:
                        pickle.dump(self,f)
                os.chdir('../..') 

    # def PLOT_MCMC_STATISTICS(self,  save_plot= True, mcmc_types_to_plot = 'all' , statistic_to_plot:str= 'acceptance_prob'):
    
    #     ## plotting
    #     plt.figure(1,figsize=(20,15))
        
    #     if mcmc_types_to_plot == 'all': 
    #         mcmc_types_to_plot = self.processed_data['MAGNETISATION'].keys()

    #     dim1 = int(len(self.data.keys()) / 2); dim2 = 2
    #     for seed in tqdm(self.seeds):
    #         for mcmc_type in mcmc_types_to_plot:
    #             stat_to_plot = self.processed_data['MCMC-STATISTICS'][seed][mcmc_type][statistic_to_plot] 
                
    #             plt.subplot(dim1,dim2,seed)

    #             if statistic_to_plot == 'acceptance_prob':
    #                 plt.hist(np.log10(stat_to_plot),
    #                     label= mcmc_type ,alpha= 0.5, 
    #                     bins= 50,density=True)
                    
    #                 if seed==9 or seed==10:
    #                     plt.xlabel("log(Acceptance Rate)")
    #             else : 
    #                 plt.hist(stat_to_plot,
    #                     label= mcmc_type ,alpha= 0.1, 
    #                     bins= 50,density=True)
                    
    #                 if seed==9 or seed==10:
    #                     plt.xlabel(statistic_to_plot)
        
    #         plt.legend()
            
    #     if save_plot:
    #         os.chdir(self.savefile_path)
    #         figname  = self.name + 'ACCEPTANCEPROB.pdf'
    #         plt.savefig(figname)
    #         os.chdir('../..')

    #         plt.show()
        
    def CALCULATE_MCMC_STATISTICS(self, save_data: bool = False, allow_reprocessing= False, seeds_new = None, mcmc_types_new= None ):

        not_computed = False
        ## data processing ##        
        if 'MCMC-STATISTICS' not in self.processed_data:
            not_computed = True
            seeds = self.seeds
            mcmc_types = self.mcmc_types
            self.processed_data['MCMC-STATISTICS'] = { seed: {} for seed in seeds }
        
        if allow_reprocessing :
            if seeds_new == None: seeds = self.seeds
            else: 
                seeds = seeds_new
                for seed in seeds:
                    self.processed_data['MCMC-STATISTICS'][seed] = {}
            
            if mcmc_types_new == None: mcmc_types = self.mcmc_types 
            else: mcmc_types = mcmc_types_new
        
        if not_computed or allow_reprocessing :
            
            for seed in tqdm(seeds, desc= "Processing MCMC Statistics"):
                for mcmc_type in mcmc_types:
                    self.processed_data['MCMC-STATISTICS'][seed][mcmc_type] = {}
                    stat_data = get_trajectory_statistics(self.data[seed][mcmc_type], self.model)
                    for stat in stat_data.keys():
                        self.processed_data['MCMC-STATISTICS'][seed][mcmc_type][stat] = stat_data[stat]
            
            if save_data:
                os.chdir(self.savefile_path)

                with open(self.name + '.pkl', 'wb') as f:
                    pickle.dump(self, f)
                
                os.chdir('../..')


## HELPER FUNCTIONS ###

def PLOT_MCMC_STATISTICS(self: ProcessMCMCData,  save_plot= False, mcmc_types_to_plot = 'all' , statistic_to_plot:str= 'acceptance_prob', kwargs_hamming = {'type': ['total', 'accepted'], 'width': 0.13}, kwargs_acceptance_prob= {'histtype':'stepfilled', 'stacked': True, 'density': True}):
    
        ## plotting
        plt.figure(1,figsize=(20,15))
        
        if mcmc_types_to_plot == 'all': 
            mcmc_types_to_plot = list(self.processed_data['MAGNETISATION'].keys())

        dim1 = int(len(self.data.keys()) / 2); dim2 = 2
        
        
        if statistic_to_plot == 'acceptance_prob':
            mcmc_labels = [mcmc_type for mcmc_type in mcmc_types_to_plot]
            pos =1
            for seed in tqdm(self.seeds):    
                
                    # stat_to_plot = [ np.log10( self.processed_data['MCMC-STATISTICS'][seed][mcmc_type][statistic_to_plot] ) for mcmc_type in mcmc_types_to_plot]
                    
                    plt.subplot(dim1,dim2,pos) ; pos += 1

                    # for mcmc_label in mcmc_labels:
                    #     stat_to_plot = np.log10( self.processed_data['MCMC-STATISTICS'][seed][mcmc_label][statistic_to_plot] )
                    #     x, bins, p = plt.hist(stat_to_plot,
                    #         label= mcmc_label ,alpha= 0.5, 
                    #         bins= np.linspace(-5,0,50), density= kwargs_acceptance_prob['density'],  histtype= kwargs_acceptance_prob['histtype'])
                    #     # print(sum(x))
                    #     # for item in p:
                    #             # item.set_height(item.get_height()/sum(x))
                    #     # print(p)
                    
                    max_ht = 0
                    for mcmc_label in mcmc_labels:
                        stat_to_plot = np.log10( self.processed_data['MCMC-STATISTICS'][seed][mcmc_label][statistic_to_plot] )
                        heights, bins = np.histogram(stat_to_plot, bins= np.linspace(-5,0,75) , density= True,)
                        heights = heights/sum(heights)
                        bin_centers = 0.5*(bins[1:] + bins[:-1])
                        bin_widths = np.diff(bins)
                        
                        plt.bar(bin_centers, heights, width=bin_widths, alpha=0.5, label = mcmc_label)    

                        if max(heights) > max_ht :
                             max_ht = max(heights)

                    if seed==9 or seed==10:
                        plt.xlabel("log(Acceptance Rate)")
                    
                    lgnd = plt.legend(loc='upper left', ncols= 4)
            
                    plt.ylim((0, max_ht + 0.05 ))        
            plt.show()        
                
        elif statistic_to_plot == 'hamming':
            mcmc_labels = [mcmc_type for mcmc_type in mcmc_types_to_plot]
            pos = 1
            for seed in tqdm(self.seeds):
                
                ticks = list(self.processed_data['MCMC-STATISTICS'][seed][mcmc_types_to_plot[0]][statistic_to_plot].keys())
                
            
                plt.subplot(dim1,dim2,pos) ; pos += 1

                width = kwargs_hamming['width']  
                x = np.arange(len(ticks))

                if 'total' in kwargs_hamming['type']:        
                    multiplier = 0            
                    for mcmc_type in mcmc_labels :
                        offset = width * multiplier
                        values_1 = [self.processed_data['MCMC-STATISTICS'][seed][mcmc_type][statistic_to_plot][key]['total'] for key in ticks ]
                        rects = plt.bar(x + offset, values_1, width, label= mcmc_type, alpha = 0.5, edgecolor = 'k')
                        # plt.bar_label(rects, padding=3)
                        multiplier += 1
                
                if 'accepted' in kwargs_hamming['type']:
                    multiplier = 0            
                    for mcmc_type in mcmc_labels :
                        offset = width * multiplier
                        values_2 = [self.processed_data['MCMC-STATISTICS'][seed][mcmc_type][statistic_to_plot][key]['accepted'] for key in ticks ]
                        if 'total' in kwargs_hamming['type']: rects = plt.bar(x + offset, values_2, width, alpha = 1.0, fill= False, edgecolor = 'k', hatch= '///')
                        else : rects = plt.bar(x + offset, values_2, width, alpha = 0.5, edgecolor = 'k', label= mcmc_type)
                        # plt.bar_label(rects, padding=3)
                        multiplier += 1
        
         
                plt.xticks(x + width, ticks)    
                if seed==9 or seed==10:
                    plt.xlabel(statistic_to_plot)
                
                lgnd = plt.legend(loc='upper left', ncols= len(mcmc_labels))
            plt.show()
                # print(lgnd.get_han())
        else :
            for seed in tqdm(self.seeds):        
                    plt.subplot(dim1,dim2,seed)

                    plt.hist(stat_to_plot,
                        label= mcmc_type ,alpha= 0.1, 
                        bins= 50,density=True)
                    
                    if seed==9 or seed==10:
                        plt.xlabel(statistic_to_plot)
            
            plt.legend(loc='upper left', ncols=3)
            
        
            
        if save_plot:
            os.chdir(self.savefile_path)
            figname  = self.name + '_' + statistic_to_plot
            plt.savefig(figname)
            os.chdir('../..')

        
            
def PLOT_MAGNETISATION(self: ProcessMCMCData, save_plot= False , mcmc_types_to_plot = 'all'):
    
        ## plotting
        fig, ax1 = plt.subplots(figsize=(26,16))
        left, bottom, width, height = [0.55, 0.2, 0.25, 0.25]

        anc0 = list(self.data.keys())[0]
        anc1 = list(self.data[anc0].keys())[0]
        dim0 = len(self.data[anc0][anc1].markov_chain)
        x=list(range(0,dim0-1))
        # mcmc_types = self.data[1].keys()
        if mcmc_types_to_plot == 'all': 
            mcmc_types_to_plot = self.processed_data['MAGNETISATION'].keys()
        
        for mcmc_type in mcmc_types_to_plot:
            
            mean_mag_local=np.mean(self.processed_data['MAGNETISATION'][mcmc_type],axis=0)
            std_local=np.std(self.processed_data['MAGNETISATION'][mcmc_type],axis=0)
            ax1.plot(x,mean_mag_local,label= mcmc_type)
            ax1.fill_between(x,mean_mag_local-std_local/2,mean_mag_local+std_local/2,alpha=0.45)


        ax1.axhline(self.model.get_observable_expectation(magnetization_of_state), label= 'actual',linestyle='--')
        ax1.legend()
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Magnetisation")
        
        if save_plot:       
                os.chdir(self.savefile_path)         
                figname = self.name + 'MAGNETISATION.pdf'
                plt.savefig(figname)
                os.chdir('../..')        
            
        plt.show()

def PLOT_KL_DIV(self:ProcessMCMCData , save_plot = False, mcmc_types_to_plot = 'all'):
    
        ## plotting
        anc0 = list(self.data.keys())[0]
        anc1 = list(self.data[anc0].keys())[0]
        dim0 = len(self.data[anc0][anc1].markov_chain)
        x=list(range(0,dim0))
        plt.figure(figsize=(12,8))
        if mcmc_types_to_plot == 'all': 
            mcmc_types_to_plot = self.processed_data['KL-DIV'].keys()
        for mcmc_type in mcmc_types_to_plot:
            plot_with_error_band(x, self.processed_data['KL-DIV'][mcmc_type] ,label= mcmc_type)
    
        plt.xlabel("iterations ")
        plt.ylabel("KL divergence")
        plt.yscale('log')
        plt.legend()

        if save_plot:
                os.chdir(self.savefile_path)      
                figname = self.name + 'KL-DIV.pdf'
                plt.savefig(figname)
        
                os.chdir('../..')        
        
        plt.show()