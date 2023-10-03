## import essential modules 
import qumcmc 
from qumcmc.basic_utils import *
# from qumcmc.energy_models import IsingEnergyFunction
from qumcmc.energy_models import IsingEnergyFunction, Exact_Sampling, random_ising_model

# from qumcmc.classical_mcmc_routines import classical_mcmc
#from qumcmc.quantum_mcmc_routines_qulacs import quantum_enhanced_mcmc     #for Qulacs Simulator backend (** Faster )
# from qumcmc.quantum_mcmc_routines_qulacs_exact import quantum_mcmc_exact
#  from qumcmc.quantum_mcmc_routines_qiskit import quantum_enhanced_mcmc   #for qiskit Aer's Simulator backend 
from qumcmc.prob_dist import DiscreteProbabilityDistribution, kl_divergence, vectoried_KL, js_divergence
from qumcmc.training import *

# from qumcmc.trajectory_processing import calculate_running_js_divergence, calculate_running_kl_divergence, calculate_runnning_magnetisation, get_trajectory_statistics, PLOT_KL_DIV, PLOT_MAGNETISATION, PLOT_MCMC_STATISTICS, ProcessMCMCData
# from scipy.linalg import expm
# from qulacs.gate import DenseMatrix, SparseMatrix
# from qulacs import QuantumState
# from qumcmc.quantum_mcmc_routines_qulacs import quantum_enhanced_mcmc
# from qumcmc.quantum_mcmc_qulacs_2 import quantum_enhanced_mcmc_2
# from qumcmc.quantum_mcmc_routines_qulacs_exact import quantum_mcmc_exact

# from itertools import permutations, product, combinations
import pickle
import os 


gridsize=3
bas=bas_dataset(grid_size=gridsize)
bas.dataset.sort()
DATA_b3 = DiscreteProbabilityDistribution(Counter( bas.horizontal_bars()  ))

nspin = len(list(DATA_b3.keys())[0])
param_model = random_ising_model(nspin, 9671032, print_model= False)
beta_train = 1.0

MCMC_SETTINGS = {
                'qu-wt1': {'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['random', 1]] , [] ] },
                'qu-wt3':{'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['random', 3]] , [] ] },
                'qu-wt2':{'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['random', 2]] , [] ] },
                'qu-alt-wt1-wt3' : {'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['random', 3], ['random', 1]] , [0.5, 0.5] ] },
                'qu-stabilizers-wt3': {'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['custom', [[0,1,2], [3,4,5], [6,7,8]] ]], []] },
                'cl-uniform': {'mcmc_type': 'classical' , 'mixer' : [  [['uniform']], [] ] }, 
                'cl-local-wt3': {'mcmc_type': 'classical' , 'mixer' : [  [['local', 3]], [] ] },
                'cl-alt-wt1-wt3': {'mcmc_type': 'classical' , 'mixer' : [  [['local', 3], ['local', 1]], [0.5, 0.5] ] } 
                }



## EXPERIMENT ##

## mcmc types 
mcmc_types = ['qu-wt1', 'qu-wt2', 'qu-wt3', 'qu-stabilizers-wt3', 'cl-uniform', 'cl-local-wt3' ]

name = "TrainingExperiments/BAS3/TRAININGEXPERIMENT_2_BAS3.pkl"

## to reinitiate experiment
TRAININGEXPERIMENT_1 = {mcmc_type: cd_training(param_model, beta_train, DATA_b3, name= mcmc_type) for mcmc_type in mcmc_types }

## to continue previous experiment
# with open(name, 'rb') as f: TRAININGEXPERIMENT_1 = pickle.load(f)
# for key in mcmc_types:
#     if key not in TRAININGEXPERIMENT_1.keys():
#         TRAININGEXPERIMENT_1[key] = cd_training(param_model, beta_train, DATA_b3, name= key)



## run experiment
EPOCHS = 300
MCMC_STEPS = 1000
lr = 0.02

for mcmc_type in mcmc_types :
    TRAININGEXPERIMENT_1[mcmc_type].train(lr =lr, mcmc_settings= MCMC_SETTINGS[mcmc_type], epochs= EPOCHS, mcmc_steps= MCMC_STEPS, verbose= True, save_training_data = {'kl_div': [True , 'last-mcmc-chain' ], 'max-min-gradient': True } , update_strategy= ['random', {'num_random_bias': 9, 'num_random_interactions': 27}])
    with open(name, 'wb') as f:  pickle.dump(TRAININGEXPERIMENT_1, f)
    