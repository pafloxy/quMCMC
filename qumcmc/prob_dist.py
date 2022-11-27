###########################################################################################
## IMPORTS ##
###########################################################################################

from basic_utils import *
from typing import Dict

class DiscreteProbabilityDistribution(dict):
    """ A class for handling discrete probability distributions """

    def __init__(self, distribution:dict) -> None :

        super().__init__(distribution) 

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
        #TODO: A faster implementation is possible by using numpy 
        len_dict = len(self)
        temp_list = [
            self[j] * dict_observable_val_at_states[j]
            for j in (list(self.keys()))
        ]
        avg = np.sum(
            temp_list
        )  # earlier I had np.mean here , which is wrong (obviously! duh!)
        return avg

