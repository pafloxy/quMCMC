import qumcmc
# from qumcmc.classical_mcmc_routines import *
## import essential modules 
from qumcmc.basic_utils import *
from qumcmc.energy_models import IsingEnergyFunction, Exact_Sampling

from qumcmc.classical_mcmc_routines import classical_mcmc 
from qumcmc.quantum_mcmc_routines import quantum_enhanced_mcmc # Manuel's code
# from qumcmc.trajectory_processing import calculate_running_js_divergence, calculate_running_kl_divergence, calculate_runnning_magnetisation, get_trajectory_statistics
from qumcmc.prob_dist import DiscreteProbabilityDistribution
### let's check if the new code is even working as desired or not
from qumcmc.mixers import * #,GenericMixer, CustomMixer, CoherentMixerSum, IncoherentMixerSum
from qumcmc.classical_mixers import *
from qumcmc.training import CDTraining
from qumcmc.mcmc_sampler_base import MCMCSampler, QuantumMCMCSampler, ClassicalMCMCSampler
######################################################################################
gridsize=3

bas= bas_dataset(grid_size=gridsize)
bas.dataset.sort()
# consider only the bars dataset and create the weight matrix for them and create the ising model
wt= hebbing_learning(bas.bas_dict["bars"]+ bas.bas_dict["stripes"])## added 2 datapoints from stripes dataset into it
n_spins=gridsize*gridsize
shape_of_J=(n_spins,n_spins)
J=-1*wt
h=np.zeros(n_spins)
model=IsingEnergyFunction(J,h,name=f'ising model BAS {n_spins}X{n_spins} bars + stripes')

# model.model_summary()
beta=1.5
## run exact sampling over all possible configurations 
exact_sampled_model = Exact_Sampling(model, beta)

## get distribution from the model
bpd_2= DiscreteProbabilityDistribution(exact_sampled_model.boltzmann_pd)
exact_sampled_model.sampling_summary()
#######################################################################################



num_chains=10
steps =10000 # 30000
gamma_range=(0.4,0.6)

initial_state="111000111"


### Creating the mixers we need

generic_mixer = GenericMixer.OneBodyMixer(n_spins)
custom_mixer_stripes = CustomMixer(n_spins,[[0,3,6],[1,4,7],[2,5,8]])
custom_mixer_bars =  CustomMixer(n_spins,[[0, 1, 2], [3, 4, 5], [6, 7, 8]] )

custom_mixer_BAS = CustomMixer(n_spins,[[0, 1, 2], [3, 4, 5], [6, 7, 8], [0,3,6] , [1,4,7] , [2,5,8]] )

coherent_mixer_only_bars = CoherentMixerSum([generic_mixer, custom_mixer_bars], 
                                            weights=[0.75, 0.25])
coherent_mixer_with_BAS_info = CoherentMixerSum([generic_mixer, custom_mixer_bars, custom_mixer_stripes],
                                                weights=[0.7,0.15,0.15])

coherent_mixer_BAS = lambda p : CoherentMixerSum([generic_mixer, custom_mixer_BAS], weights= [1-p, p] )
incoherent_mixer_BAS = lambda p : IncoherentMixerSum([generic_mixer, custom_mixer_BAS], probabilities= [1-p, p] )



incoherent_mixer_only_bars = IncoherentMixerSum([generic_mixer, custom_mixer_bars],
                                                probabilities=[0.75, 0.25])
incoherent_mixer_with_BAS_info = IncoherentMixerSum([generic_mixer, custom_mixer_bars, 
                                                    custom_mixer_stripes],
                                                probabilities=[0.7,0.15,0.15])


unfrm_clmixer= UniformProposals(n_spins)
fxdwt_clmixer= FixedWeightProposals(n_spins, 4)
cstm_clmixer= CustomProposals(n_spins, flip_pattern= [0,2,3,6])

combined_clmixer= CombineProposals([unfrm_clmixer, fxdwt_clmixer, cstm_clmixer], probabilities= [0.6,0.1,0.3])


######################################################################################### 

qmxrs = [generic_mixer, custom_mixer_bars, custom_mixer_BAS, coherent_mixer_only_bars, coherent_mixer_with_BAS_info, incoherent_mixer_only_bars, incoherent_mixer_with_BAS_info]
cmxrs = [unfrm_clmixer, fxdwt_clmixer, cstm_clmixer, combined_clmixer]

random_model = IsingEnergyFunction(np.random.randn(n_spins, n_spins), np.random.randn(n_spins), name = f'RandomModel|N={n_spins}' )

training_inst = CDTraining(random_model, 1.0, bpd_2, name = "Train")

# training_inst.train()

#################################################################

# for mxr in qmxrs+cmxrs : 
#     print(mxr)
#     print("--------------------------- \n")

# print("--------------------------- \n")
for mxr in qmxrs :
    print(f"testing {mxr} ------------")     
    quantum_mcmc = QuantumMCMCSampler(10, model, mxr, 0.5, initial_state=initial_state, verbose= True)
    quantum_mcmc.run()
    print(f" MCMC run : OK ------------")     
    quantum_mcmc.verbose = False 
    training_inst.train(quantum_mcmc, 10)
    print(f" Training run : OK ------------")
    # print(f"{mxr}: OK \n ----------------")
print("--------------------------- \n")
for mxr in cmxrs : 
    print(f"testing {mxr} ------------")     
    clmcmc = ClassicalMCMCSampler(10, model, mxr, initial_state=initial_state, verbose= True)
    clmcmc.run()
    print(f" MCMC run : OK ------------")   
    clmcmc.verbose = False   
    training_inst.train(clmcmc, 10)
    print(f" Training run : OK ------------")
    # print(f"{mxr}: OK \n ----------------")


################################################################# 
