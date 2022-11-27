###########################################################################################
## IMPORTS ##
###########################################################################################

from basic_utils import *
from typing import Dict

class DiscreteProbabilityDistribution(dict):
    """ A class for handling discrete probability distributions """

    def __init__(self, distribution:dict) -> None :

        self.dist = distribution
    pass 
    # self.issubset