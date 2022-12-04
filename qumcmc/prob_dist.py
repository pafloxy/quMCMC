###########################################################################################
## IMPORTS ##
###########################################################################################

from .basic_utils import *
from typing import Dict

class DiscreteProbabilityDistribution(dict):
    """ A class for handling discrete probability distributions """

    def __init__(self, distribution:dict) -> None :

        super().__init__(distribution) 
    
    def normalise(self, print_normalisation:bool= False):
        r_sum = np.sum(list(self.values()))
        for k in list(self.keys()) :
            self[k] = self[k] / r_sum

    def value_sorted_dict(self, reverse=False):
        """ Sort the dictionary in ascending or descending(if reverse=True) order of values. """
        sorted_dict = {
            k: v
            for k, v in sorted(self.items(), key=lambda item: item[1], reverse=reverse)
        }
        return sorted_dict

    def index_sorted_dict(self, reverse= False):
        """ Sort the dictionary in ascending or descending(if reverse=True) order of index. """
        sorted_dict = {
            k: v
            for k, v in sorted(self.items(), key=lambda item: item[0], reverse=reverse)
        }
        return sorted_dict
        
    def expectation(self, dict_observable_val_at_states: dict):
        """
        new version:
        Returns average of any observable of interest

        Args:
        self= {state: probability}
        dict_observable_val_at_states={state (same as that of self): observable's value at that state}

        Returns:
        avg
        """
        ##TODO: A faster implementation is possible by using numpy 
        len_dict = len(self)
        temp_list = [
            self[j] * dict_observable_val_at_states[j]
            for j in (list(self.keys()))
        ]
        avg = np.sum(
            temp_list
        )  # earlier I had np.mean here , which is wrong (obviously! duh!)
        return avg
        
### SOME FUNCTIONS.
#1. KL divergence
#2. JS divergence

def kl_divergence(dict_p:dict,dict_q:dict, prelim_check=True):
    ''' 
    Returns KL divergence KL(p||q);

    Args:

    dict_p: distribution p ({random_variable: prob}),

    dict_q: distribution q ({random_variable: prob}),

    prelim_check: default 'True'. 
    If user is completely sure that 
    dict_p and dict_q have same keys and that both the distributions are
    normalised then user can set it to 'False'.
    '''
    if prelim_check:
        #check for whether or not dict_p and dict_q have same keys
        keys_p,keys_q=list(dict_p.keys()),list(dict_q.keys())# reason why not inplace
        keys_p.sort();keys_q.sort()
        assert keys_p==keys_q, "keys of both the dictionaries dont match!"

        # check for whether values add to 1.
        eps=1e-6
        sum_vals_p=np.sum(list(dict_p.values()))
        assert np.abs(sum_vals_p-1.0)<=eps, "sum of values of dict_p must be 1."
        sum_vals_q=np.sum(list(dict_q.values()))
        assert np.abs(sum_vals_q-1.0)<=eps, "sum of values of dict_q must be 1."
    
    #prep for caln
    p=DiscreteProbabilityDistribution(dict_p).index_sorted_dict()
    q=DiscreteProbabilityDistribution(dict_q).index_sorted_dict()
    p_arr,q_arr=np.array(list(p.values())).reshape((len(p))), np.array(list(q.values())).reshape((len(q)))
    return np.sum(np.where(p_arr>10**-6,p_arr*np.log2(p_arr/q_arr),0.))

def js_divergence(dict_p:dict,dict_q:dict, prelim_check=True):
    ''' 
    Returns JS divergence JS(p||q);
    
    Args:
    dict_p: distribution p ({random_variable: prob}),

    dict_q: distribution q ({random_variable: prob}),

    prelim_check: default 'True'. 
    If user is completely sure that 
    dict_p and dict_q have same keys and that both the distributions are
    normalised then user can set it to 'False'.
    '''
    if prelim_check:
        #check for whether or not dict_p and dict_q have same keys
        keys_p,keys_q=list(dict_p.keys()),list(dict_q.keys())
        keys_p.sort();keys_q.sort()
        assert keys_p==keys_q, "keys of both the dictionaries dont match!"
        
        # check for whether values add to 1.
        eps=1e-6
        sum_vals_p=np.sum(list(dict_p.values()))
        assert np.abs(sum_vals_p-1.0)<=eps, "sum of values of dict_p must be 1."
        sum_vals_q=np.sum(list(dict_q.values()))
        assert np.abs(sum_vals_q-1.0)<=eps, "sum of values of dict_q must be 1."

    #prep for caln
    p=DiscreteProbabilityDistribution(dict_p).index_sorted_dict()
    q=DiscreteProbabilityDistribution(dict_q).index_sorted_dict()
    p_arr,q_arr=np.array(list(p.values())).reshape((len(p))), np.array(list(q.values())).reshape((len(q)))
    val_m = np.round(0.5 * (p_arr + q_arr),decimals=8)
    #print("val_m:");print(val_m)
    m=dict(zip(list(p.keys()),val_m))
    #print("m:");print(m)
    return 0.5 * (kl_divergence(p, m, prelim_check=False) +  kl_divergence(q, m, prelim_check=False))
