## import essential modules ##
from qumcmc.basic_utils import *
from qumcmc.energy_models import *
from qumcmc.classical_mcmc_routines import *
from qumcmc.quantum_mcmc_routines_qulacs import *     #for Qulacs Simulator backend
# from qumcmc.quantum_mcmc_routines_qulacs import quantum_enhanced_mcmc   #for qiskit Aer's Simulator backend 
from qumcmc.trajectory_processing import *
from qumcmc.training import *

#############
import pandas as pd
#############



def int_to_str(state_obtained, nspin):
    return f"{state_obtained:0{nspin}b}"

def run_sampling_task(nspin, seed, beta, steps):
    
    model = random_ising_model(nspin, seed, print_model=False)
    model_ems =Exact_Sampling(model, beta= beta)
    initial = model_ems.boltzmann_pd.get_sample(1)[0]
    ## run mcmc ##
    clchain =classical_mcmc(
        n_hops=steps,
        model=model,
        temperature=1/beta,
        initial_state=initial
    )
    qchain =quantum_enhanced_mcmc(
        n_hops=steps,
        model=model,
        temperature=1/beta,
        initial_state=initial
    )
    ## save data ##
    trajectory_data = {};trajectory_data['model_seed'] = seed; trajectory_data['beta'] = beta
    trajectory_data['classical'] = get_trajectory_statistics(clchain, model_ems, to_observe={'acceptance_prob', 'kldiv'})
    trajectory_data['quantum'] =  get_trajectory_statistics(qchain, model_ems, to_observe={'acceptance_prob', 'kldiv'})

    return trajectory_data


######## Initialisations ###########
save_after_run = False
####################################

####################################
###     nspin: 5  ###
####################################
DATA_5qubit = {};nspin = 5
seeds5q = [23564, 202687, 742410, 156407, 501064]
betas5q = [1.0134]*5
iter = 0
for seed, beta in tqdm(zip(seeds5q, betas5q)):
    iter += 1
    tdata = run_sampling_task(nspin, seed, beta, 5000)
    DATA_5qubit[iter] = tdata

    if save_after_run :
        to_save = pd.DataFrame(tdata)
        to_save.to_json("SamplingData/DATA_5qubit_"+str(iter)+".json")
    
DATA_5qubit = pd.DataFrame(DATA_5qubit)
DATA_5qubit.to_json("SamplingData/DATA_5qubit.json")


#####################################
###     nspin: 10 ###
#####################################
DATA_10qubit = {}; nspin = 10
seeds10q = [23564,178064,32164,143264,13164]
betas10q = [1.0134,1.0134,1.0134,1.5634,1.12134]
iter = 0
for seed, beta in tqdm(zip(seeds10q, betas10q)):
    iter += 1
    tdata = run_sampling_task(nspin, seed, beta, 10000)
    DATA_10qubit[iter] = tdata
    
    if save_after_run :
        to_save = pd.DataFrame(tdata)
        to_save.to_json("SamplingData/DATA_10qubit_"+str(iter)+".json")

DATA_10qubit = pd.DataFrame(DATA_10qubit)
DATA_10qubit.to_csv("SamplingData/DATA_10qubit.json")

#####################################
###     nspin: 15 ###
#####################################
# DATA_15qubit = {}; nspin = 15
# seeds15q = [23564, 40217, 4036997, 98997, 14797]
# betas15q = [1.02834, 1.02834, 1.02834, 1.02834, 1.02834]
# iter = 1
# for seed, beta in tqdm(zip(seeds15q[iter:], betas15q[iter:])):
#     iter += 1
#     tdata = run_sampling_task(nspin, seed, beta, 10000)
#     DATA_15qubit[iter] = tdata
    
#     if save_after_run :
#         to_save = pd.DataFrame(tdata)
#         to_save.to_json("SamplingData/DATA_15qubit_"+str(iter)+".json")
    
# DATA_15qubit = pd.DataFrame(DATA_15qubit)
# DATA_15qubit.to_json("SamplingData/DATA_15qubit.json")

# ####################################
# ##     nspin: 20 ###
# ####################################
# DATA_20qubit = {}; nspin = 20
# seeds20q = [54797, 7497, 4917, 49178, 9178]
# betas20q = [1.02834, 1.02834, 1.02834, 1.02834, 1.02834]
# iter = 0
# for seed, beta in tqdm(zip(seeds20q, betas20q)):
#     iter += 1
#     tdata = run_sampling_task(nspin, seed, beta, 10000)
#     DATA_20qubit[iter] = tdata
        
#     if save_after_run :
#         to_save = pd.DataFrame(tdata)
#         to_save.to_csv("SamplingData/DATA_20qubit_"+str(iter)+".csv")

# DATA_20qubit = pd.DataFrame(DATA_20qubit)
# DATA_20qubit.to_csv("SamplingData/DATA_20qubit.csv")