###########################################################################################
## IMPORTS ##
###########################################################################################

from .basic_utils import *
from typing import Dict
from copy import deepcopy
import pandas as pd
from typing import Union
import seaborn as sns
import random
import pickle as pkl

# import
from .basic_utils import *
from .prob_dist import DiscreteProbabilityDistribution, kl_divergence, vectoried_KL
from .energy_models import IsingEnergyFunction, Exact_Sampling, random_ising_model
from .mcmc_sampler_base import MCMCSampler, QuantumMCMCSampler, ClassicalMCMCSampler

# from .classical_mcmc_routines import classical_mcmc
# from .quantum_mcmc_routines import  quantum_enhanced_mcmc  # for qulacs Simulator backend


# from .quantum_mcmc_routines_qulacs import quantum_enhanced_mcmc   #for qiskit Aer's Simulator backend
# from .trajectory_processing import (
#     calculate_running_kl_divergence,
#     calculate_runnning_magnetisation,
#     get_trajectory_statistics,
# )

# from qulacs import QuantumState
# from qulacs_core import DensityMatrix

# from qiskit.visualization import plot_histogram

###########################################################################################
## HELPER FUNCTIONS ##
###########################################################################################

int_to_binary = lambda state_obtained, n_spins: f"{state_obtained:0{n_spins}b}"
binary_to_bipolar = lambda string: 2.0 * float(string) - 1.0


def get_observable_expectation(
    observable: callable, mcmc_chain: MCMCChain, skip_init: int = 100
):

    sample_observable = []
    for s in mcmc_chain.accepted_states:

        sample_observable.append(observable(s))

    sample_observable = np.array(sample_observable)

    return sample_observable.mean(dtype=float)  # , sample_observable.var(dtype= float)


def correlation_spins(state: str, indices: Union[tuple, List]):

    assert len(indices) <= len(state)

    prod = 1.0
    for j in indices:
        prod *= binary_to_bipolar(state[j])

    return prod


def cd_J(
    index, data_distribution: DiscreteProbabilityDistribution, mcmc_chain: MCMCChain
):

    assert len(index) == 2
    observable = lambda s: correlation_spins(s, [index[0], index[1]])
    r = data_distribution.get_observable_expectation(
        observable
    ) - get_observable_expectation(observable, mcmc_chain, skip_init=100)

    return r


def cd_h(
    index: int,
    data_distribution: DiscreteProbabilityDistribution,
    mcmc_chain: MCMCChain,
):

    assert isinstance(index, int)
    observable = lambda s: correlation_spins(s, [index])
    r = data_distribution.get_observable_expectation(
        observable
    ) - get_observable_expectation(observable, mcmc_chain, skip_init=100)

    return r


###########################################################################################
## MAIN TRAINING CLASS ##
###########################################################################################


# @dataclass
class CDTraining:
    """
    model: initial model = (J init, h init) at some temp T
    beta: 1/Temperature
    data_dist: empirical data which we want to learn!
    name: name of the training instance

    """

    def __init__(
        self,
        model: IsingEnergyFunction,
        beta: float,
        data_dist: DiscreteProbabilityDistribution,
        name: str = "train",
        pickle_loc: Union[None, str] = None
    ) -> None:
        self.model = deepcopy(model)
        self.model_beta = beta
        self.data_distribution = data_dist
        self.training_history = {
            "specifications": [],
            "kl_div": [],
            "max-min-gradient": [],
        }
        self.kl_div = []
        self.list_pair_of_indices = [
            [i, j]
            for i in range(1, self.model.num_spins)
            for j in range(i, self.model.num_spins)
            if j != i
        ]
        self.name = name
        self.pickle_fileloc = pickle_loc

    def cd_J(self, index, mcmc_chain: MCMCChain):

        assert len(index) == 2
        observable = lambda s: correlation_spins(s, [index[0], index[1]])
        r = self.data_distribution.get_observable_expectation(
            observable
        ) - get_observable_expectation(observable, mcmc_chain, skip_init=100)

        return r

    def cd_h(self, index: int, mcmc_chain: MCMCChain):

        assert isinstance(index, int)
        observable = lambda s: correlation_spins(s, [index])
        r = self.data_distribution.get_observable_expectation(
            observable
        ) - get_observable_expectation(observable, mcmc_chain, skip_init=100)

        return r

    # @setattr
    # def data_distribution()

    def _train_on_mcmc_chain(
        self,
        mcmc_sampler: Union[ClassicalMCMCSampler, QuantumMCMCSampler],
        lr: float = 0.01,
        mcmc_steps: int = 1000,
        update_strategy=["all", []],
        save_gradient_info=False,
    ):  # we will try to increase mcmc steps.

        # random.seed(random.random()) ## add seed to random ##TODO
        initial_state = self.data_distribution.get_sample(1)[
            0
        ]  ##randomly select a state from the data distribution
        mcmc_sampler.n_hops = mcmc_steps
        mcmc_sampler.initial_state = initial_state
        mcmc_sampler.model = self.model
        
        self.mcmc_chain = mcmc_sampler.run()

        max_grad_h = 0
        max_grad_J = 0
        min_grad_h = 0
        min_grad_J = 0
        if update_strategy[0] == "random":  ## random update strategy ##

            ## just realised that even this is not a good thing!

            assert (
                update_strategy[1]["num_random_bias"] <= self.model.num_spins
            ), f"update_strategy[1]['num_random_bias'] should be <= num_spins (which is= {self.model.num_spins}) "
            assert update_strategy[1]["num_random_interactions"] <= len(
                self.list_pair_of_indices
            ), f"update_strategy[1]['num_random_interactions'] should be <=len(self.list_pair_of_indices) (which is= {len(self.list_pair_of_indices)})"

            list_random_indices = random.sample(
                range(0, self.model.num_spins), update_strategy[1]["num_random_bias"]
            )

            list_pair_of_different_indices = [
                [
                    list_random_indices[j],
                    random.choice(
                        list(range(0, list_random_indices[j]))
                        + list(range(list_random_indices[j] + 1, self.model.num_spins))
                    ),
                ]
                for j in range(0, update_strategy[1]["num_random_bias"])
            ]

            ## Update J
            for k in range(len(list_pair_of_different_indices)):

                indices_J = list_pair_of_different_indices[k]
                calculate_cd_J = self.cd_J(indices_J, self.mcmc_chain)
                updated_param_j = (
                    self.model.J[indices_J[0], indices_J[1]] - lr * calculate_cd_J
                )
                self.model._update_J(updated_param_j, indices_J)

                if save_gradient_info:
                    if k == 0:
                        max_grad_J = np.abs(calculate_cd_J)
                        min_grad_J = np.abs(calculate_cd_J)
                    if np.abs(calculate_cd_J) > max_grad_J:
                        max_grad_J = np.abs(calculate_cd_J)
                    if np.abs(calculate_cd_J) < min_grad_J:
                        min_grad_J = np.abs(calculate_cd_J)

            ## Update h
            for k in range(update_strategy[1]["num_random_bias"]):

                index_h = list_random_indices[k]
                calculate_cd_h = self.cd_h(index_h, self.mcmc_chain)
                updated_param_h = self.model.h[index_h] - lr * calculate_cd_h
                self.model._update_h(updated_param_h, index_h)

                if save_gradient_info:
                    if k == 0:
                        max_grad_h = np.abs(calculate_cd_h)
                        min_grad_h = np.abs(calculate_cd_h)
                    if np.abs(calculate_cd_h) > max_grad_h:
                        max_grad_h = np.abs(calculate_cd_h)
                    if np.abs(calculate_cd_h) < min_grad_h:
                        min_grad_h = np.abs(calculate_cd_h)

            if save_gradient_info:
                self.training_history["max-min-gradient"].append(
                    [(max_grad_J, min_grad_J), (max_grad_h, min_grad_h)]
                )

        if update_strategy[0] == "all":

            for i in range(0, self.model.num_spins):
                for j in range(0, i):

                    calculate_cd_J = self.cd_J([i, j], self.mcmc_chain)
                    updated_param_j = self.model.J[i, j] - lr * calculate_cd_J
                    self.model._update_J(updated_param_j, [i, j])
                    if save_gradient_info:
                        if i + j == 0:
                            max_grad_J = np.abs(calculate_cd_J)
                            min_grad_J = np.abs(calculate_cd_J)
                        if np.abs(calculate_cd_J) > max_grad_J:
                            max_grad_J = np.abs(calculate_cd_J)
                        if np.abs(calculate_cd_J) < min_grad_J:
                            min_grad_J = np.abs(calculate_cd_J)

                calculate_cd_h = self.cd_h(i, self.mcmc_chain)
                updated_param_h = self.model.h[i] - lr * calculate_cd_h
                self.model._update_h(updated_param_h, i)
                if save_gradient_info:
                    if i == 0:
                        max_grad_h = np.abs(calculate_cd_h)
                        min_grad_h = np.abs(calculate_cd_h)
                    if np.abs(calculate_cd_h) > max_grad_h:
                        max_grad_h = np.abs(calculate_cd_h)
                    if np.abs(calculate_cd_h) < min_grad_h:
                        min_grad_h = np.abs(calculate_cd_h)

            if save_gradient_info:
                self.training_history["max-min-gradient"].append(
                    [(max_grad_J, min_grad_J), (max_grad_h, min_grad_h)]
                )

    def train(
        self,
        mcmc_sampler: Union[ClassicalMCMCSampler, QuantumMCMCSampler],
        mcmc_steps: int = 500,
        lr: float = 0.01,
        # mcmc_settings={"mcmc_type": "quantum-enhanced", "mixer": [[["random", 1]], []]},
        epochs: int = 10,
        update_strategy=["all", []],
        save_training_data={
            "kl_div": [True, "exact-sampling"],
            "max-min-gradient": True,
        },
        verbose=True,
        save_picle:bool = False
    ):
        """mcmc_sampler : type of mcmc sampler to be used for sampling from the paramterised model
        update_strategy : chocie of the parameter update strategy
                         -> ['all' , []] : updates all paramters in the model
                         -> ['random', {'num_random_bias': , 'num_random_interactions':}] : randomly pick num_random_ias 'bias' and num_random_interaction 'interaction' paramaters
                                                                        to be updated in the call
        """
        self.training_history["specifications"].append(
            {
                "lr": lr,
                "sampler": mcmc_sampler,
                "epochs": epochs,
                "update_strategy": update_strategy,
                # "mcmc_steps": mcmc_steps,
            }
        )
        iterator = tqdm(range(epochs), desc="training epochs", disable=not verbose)
        iterator.set_postfix(
            {
                "sampler": mcmc_sampler.name,
                "update-strategy": update_strategy[0],
            }
        )
        for epoch in iterator:

            self._train_on_mcmc_chain(
                mcmc_sampler=mcmc_sampler,
                lr=lr,
                mcmc_steps=mcmc_steps,
                update_strategy=update_strategy,
                save_gradient_info=save_training_data["max-min-gradient"],
            )

            if save_training_data["kl_div"][0]:

                if save_training_data["kl_div"][1] == "last-mcmc-chain":
                    ## calculate kl-div from last mcmc_chain
                    self.training_history["kl_div"].append(
                        kl_divergence(
                            self.data_distribution,
                            self.mcmc_chain.get_accepted_dict(normalize=True),
                        )
                    )

                if save_training_data["kl_div"][1] == "exact-sampling":
                    ## calculate kl-div from Exact Sampling current model
                    exact_sampled_model = Exact_Sampling(self.model, self.model_beta)
                    self.training_history["kl_div"].append(
                        kl_divergence(
                            self.data_distribution, exact_sampled_model.boltzmann_pd
                        )
                    )

                iterator.set_postfix(
                    {
                        "sampler": mcmc_sampler.name,
                        "update-strategy": update_strategy[0],
                        "kl div ": self.training_history["kl_div"][-1],
                    }
                )
        if save_picle: 
            if self.pickle_fileloc:
                with open(self.pickle_fileloc, 'wb') as f : 
                    pkl.dump(self, f)
            else: 
                raise ValueError(f"{self.pickle_fileloc} must be .pkl file")
        ## update training data ##
        # self.kl_div += kl_div
        # self.training_history['kl_div']= self.kl_div

    ########### scheduled-training ####################
    ###################################################

    # def train(self, lr:float= 0.01, method = 'quantum-enhanced',
    # epochs:int = 10, schedule:str= 'linear', update_strategy[1]['num_random_interactions']:int=5,
    # save_training_data:bool = True ):

    #     ## random update strategy ##
    #     kl_div = []; js_div= []
    #     iterator = tqdm(range(epochs), desc= 'training epochs')
    #     iterator.set_postfix({'method': method})

    #     if schedule == 'linear':
    #         mcmc_steps = np.linspace(100, 5000, epochs, dtype= int)
    #         # params = self.model.num_spins * (self.model.num_spins + 1) / 2
    #         update_strategy[1]['num_random_bias'] = np.linspace(int(self.model.num_spins/4), self.model.num_spins, epochs, dtype= int)
    #         lr_c = np.linspace(lr, 10 * lr, epochs, dtype= float )

    #     for epoch in iterator:

    #         self._train_on_mcmc_chain(lr= lr_c[epoch] ,
    #         method = method, update_strategy[1]['num_random_bias']= update_strategy[1]['num_random_bias'][epoch], update_strategy[1]['num_random_interactions']=update_strategy[1]['num_random_interactions'],
    #         mcmc_steps= mcmc_steps[epoch] )

    #         if save_training_data:

    #             kl_div.append(kl_divergence(  self.data_distribution,self.mcmc_chain.get_accepted_dict(normalize= True)  ))
    #             iterator.set_postfix( { 'method ': method, 'js div ' : js_div[-1], 'mcmc-steps ': mcmc_steps[epoch] })

    #     self.training_history['kl_div']= kl_div
