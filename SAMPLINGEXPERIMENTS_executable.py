## import essential modules 
import qumcmc 
from qumcmc.basic_utils import *
# from qumcmc.energy_models import IsingEnergyFunction
from qumcmc.energy_models import IsingEnergyFunction, Exact_Sampling

from qumcmc.classical_mcmc_routines import classical_mcmc
#from qumcmc.quantum_mcmc_routines_qulacs import quantum_enhanced_mcmc     #for Qulacs Simulator backend (** Faster )
# from qumcmc.quantum_mcmc_routines_qulacs_exact import quantum_mcmc_exact
#  from qumcmc.quantum_mcmc_routines_qiskit import quantum_enhanced_mcmc   #for qiskit Aer's Simulator backend 

from qumcmc.trajectory_processing import calculate_running_js_divergence, calculate_running_kl_divergence, calculate_runnning_magnetisation, get_trajectory_statistics, PLOT_KL_DIV, PLOT_MAGNETISATION, PLOT_MCMC_STATISTICS, ProcessMCMCData
# from scipy.linalg import expm
# from qulacs.gate import DenseMatrix, SparseMatrix
# from qulacs import QuantumState
# from qumcmc.quantum_mcmc_routines_qulacs import quantum_enhanced_mcmc
from qumcmc.quantum_mcmc_qulacs_2 import quantum_enhanced_mcmc_2
# from qumcmc.quantum_mcmc_routines_qulacs_exact import quantum_mcmc_exact
from qumcmc.prob_dist import DiscreteProbabilityDistribution

import pickle
import os 

#####################################################################
#####################################################################


gridsize=3
bas=bas_dataset(grid_size=gridsize)
bas.dataset.sort()

DATA_b3 = DiscreteProbabilityDistribution(Counter( bas.horizontal_bars()  ))

# considering only bars dataset
wt=hebbing_learning(bas.bas_dict['bars'])

# creating ising model for the bas only dataset
n_spins=gridsize*gridsize
shape_of_J=(n_spins,n_spins)
J=-1*wt
h=np.zeros(n_spins)
model=IsingEnergyFunction(J,h,name=f'ising model BAS {n_spins}X{n_spins} bars only')

beta=1.5
## run exact sampling over all possible configurations 
exact_sampled_model = Exact_Sampling(model, beta)

## get distribution from the model
bpd=exact_sampled_model.boltzmann_pd
# exact_sampled_model.sampling_summary()



#####################################################################
#####################################################################

MCMC_SETTINGS = {
                'qu-wt1': {'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['random', 1]] , [] ] },
                'qu-wt2':{'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['random', 2]] , [] ] },
                'qu-wt3':{'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['random', 3]] , [] ] },
                'qu-alt-wt1-wt3' : {'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['random', 3], ['random', 1]] , [0.5, 0.5] ] },
                'qu-stabilizers-wt3': {'mcmc_type': 'quantum-enhanced' , 'mixer' : [ [['custom', [[0,1,2], [3,4,5], [6,7,8]] ]], []] },
                'cl-uniform': {'mcmc_type': 'classical' , 'mixer' : [  [['uniform']], [] ] }, 
                'cl-local-wt1': {'mcmc_type': 'classical' , 'mixer' : [  [['local', 1]], [] ] },
                'cl-local-wt3': {'mcmc_type': 'classical' , 'mixer' : [  [['local', 3]], [] ] },
                'cl-alt-wt1-wt3': {'mcmc_type': 'classical' , 'mixer' : [  [['local', 3], ['local', 1]], [0.5, 0.5] ] } 
                }

#####################################################################
## EXPERIMENT ##
#####################################################################

## mcmc types 
# mcmc_types = ['qu-wt1', 'qu-wt2', 'qu-wt3', 'qu-stabilizers-wt3', 'cl-uniform', 'cl-local-wt3' ]
mcmc_types = ['cl-uniform', 'cl-local-wt3', 'cl-local-wt1' ]

## seeds
num_different_chains=10
mcmc_steps = 15000

## data saving location
name_data = "SamplingData/BAS3/SAMPLINGDATA_BAS3.pkl"
name_result = "SamplingData/BAS3/SAMPLINGRESULT_BAS3.pkl"

## to reinitiate experiment TODO
# SAMPLINGDATA_BAS3 = {}
# SAMPLINGRESULT_BAS3 = {}

## to continue previous experiment
with open(name_data, 'rb') as f: SAMPLINGDATA_BAS3 = pickle.load(f)
with open(name_result, 'rb') as f: SAMPLINGRESULT_BAS3 = pickle.load(f)
model = SAMPLINGRESULT_BAS3.model

for mcmc_setting in mcmc_types :

    for seed_val in tqdm(range(1,num_different_chains+1), desc= str(mcmc_setting)):

        if MCMC_SETTINGS[mcmc_setting]['mcmc_type'] == 'classical': 
            steps=mcmc_steps
            #list_labels.append('classical uniform strategy')
            mcmc_chain =classical_mcmc(
                n_hops=steps,
                model=model,
                temperature=1/beta,
                initial_state= DATA_b3.get_sample(1)[0],
                proposition_method= MCMC_SETTINGS[mcmc_setting]['mixer']       
            )
        
        if MCMC_SETTINGS[mcmc_setting]['mcmc_type'] == 'quantum-enhanced': 
            steps=mcmc_steps
            #list_labels.append('classical uniform strategy')
            mcmc_chain =quantum_enhanced_mcmc_2(
                n_hops=steps,
                model=model,
                temperature=1/beta,
                gamma_range= (0.1, 0.4),
                initial_state= DATA_b3.get_sample(1)[0],
                mixer= MCMC_SETTINGS[mcmc_setting]['mixer']       
            )
         
    SAMPLINGDATA_BAS3[seed_val][mcmc_setting] = mcmc_chain
    SAMPLINGRESULT_BAS3.UPDATE_DATA(SAMPLINGDATA_BAS3, None, [mcmc_setting], save_data= False )
    
    with open(name_data, 'wb') as f: pickle.dump(SAMPLINGDATA_BAS3, f)
    with open(name_result, 'wb') as f: pickle.dump(SAMPLINGRESULT_BAS3, f)




