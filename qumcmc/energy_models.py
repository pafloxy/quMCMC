###########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
import matplotlib.pyplot as plt
from .basic_utils import plot_bargraph_desc_order
from .prob_dist import value_sorted_dict, DiscreteProbabilityDistribution
from typing import Union
from tqdm import tqdm
import seaborn as sns

from qulacs import PauliOperator, Observable

###########################################################################################
## ENERGY MODEL ##
###########################################################################################


class IsingEnergyFunction:
    """A class to build the Ising Energy Function from self"""

    def __init__(
        self, J: np.array, h: np.array, name: str = None, circuit_J=None, circuit_h=None
    ) -> None:
        """
        ARGS:
        ----
        J: weight-matrix of the interactions between the spins
        h: local field to the spins

        """
        self.J = J
        self.h = h

        self.circuit_J = circuit_J if circuit_J is not None else J
        self.circuit_h = circuit_J if circuit_h is not None else h

        # self.beta = beta
        self.num_spins = len(h)
        # self.exact_sampling_status = False
        if np.allclose(J, 0):
            self.alpha = 1.0
        else:
            self.alpha = np.sqrt(self.num_spins) / np.sqrt(
                sum([J[i][j] ** 2 for i in range(self.num_spins) for j in range(i)])
                + sum([h[j] ** 2 for j in range(self.num_spins)])
            )

        if name == None:
            self.name = "JK_random"
        else:
            self.name = name

    @property
    def get_J(self):
        return self.J

    @property
    def get_h(self):
        return self.h

    # @property
    def model_summary(self, plot=True):

        print("=============================================")
        print("            MODEL : " + str(self.name))
        print("=============================================")

        print(
            "Non-zero Interactions (J) : "
            + str(int(np.count_nonzero(self.J) / 2))
            + " / "
            + str(int(0.5 * self.num_spins * (self.num_spins - 1)))
        )
        print(
            "Non-zero Bias (h) : "
            + str(int(np.count_nonzero(self.h)))
            + " / "
            + str(self.num_spins)
        )
        print("---------------------------------------------")

        print("Average Interaction Strength <|J|> : ", np.mean(np.abs(self.J)))
        print("Average Bias Strength <|h|>: ", np.mean(np.abs(self.h)))
        print("alpha : ", self.alpha)

        print("---------------------------------------------")

        # sns.set()
        if plot:
            plt.figure(figsize=(16, 10))
            to_plot = self.J
            for i in range(self.num_spins):
                to_plot[i][i] = self.h[i]
            sns.heatmap(to_plot, square=True, annot=False, cbar=True)

    # def get_hamiltonian
    def get_hamiltonian(self):

        J = self.get_J
        h = self.get_h

        hamiltonian = Observable(self.num_spins)

        for i in range(0, self.num_spins):

            pstr = "Z " + str(i)
            hamiltonian.add_operator(PauliOperator(pstr, coef=h[i]))

            for j in range(0, i):

                pstr = "Z " + str(i) + " " + "Z " + str(j)
                hamiltonian.add_operator(PauliOperator(pstr, coef=J[i, j]))

        return hamiltonian

    def get_energy(self, state: Union[str, np.array]) -> float:
        """Returns the energy of a given state

        ARGS:
        ----
        state : configuration of spins for which the energy is requrieed to be calculated.
                NOTE:  if input is an numpy array then it should only consist of bipolar values -> {+1, -1}

        """

        if isinstance(state, str):
            state = np.array([-1 if elem == "0" else 1 for elem in state])
            # state = np.array( [int(list(state)[i]) for i in range(len(state))])
            energy = 0.5 * np.dot(state.transpose(), self.J.dot(state)) + np.dot(
                self.h.transpose(), state
            )
            return energy
        else:
            return 0.5 * np.dot(state.transpose(), self.J.dot(state)) + np.dot(
                self.h.transpose(), state
            )

    def get_boltzmann_factor(
        self, state: Union[str, np.array], beta: float = 1.0
    ) -> float:
        """Get un-normalised boltzmann probability of a given state

        ARGS:
        ----
        state : configuration of spins for which probability is to be calculated
        beta : inverse temperature (1/T) at which the probability is to be calculated.

        """
        return np.exp(-1 * beta * self.get_energy(state))

    from typing import List

    def _update_J(self, new_param: float, index: Union[tuple, List]):

        assert len(index) == 2
        self.J[index[0], index[1]] = new_param
        self.J[index[1], index[0]] = new_param

    def _update_h(self, new_param: float, index: int):

        self.h[index] = new_param


###########################################################################################
## EXACT SAMPLING on MODEL ##
###########################################################################################


class Exact_Sampling(IsingEnergyFunction):

    def __init__(
        self, model: IsingEnergyFunction, beta: float = 1.0, verbose=False
    ) -> None:

        super().__init__(model.get_J, model.get_h, model.name)

        self.beta = beta
        self.exact_sampling_status = False
        self.run_exact_sampling(self.beta, verbose=verbose)

    def sampling_summary(
        self, plot_dist: bool = True, show_threshold=0.01, ylabel="Prob"
    ):

        if self.exact_sampling_status:
            tmp = np.array(list(self.boltzmann_pd.values()))
            count_non_zero = len(tmp[tmp > show_threshold])

            print("=============================================")
            print("     MODEL : " + str(self.name) + " |  beta : " + str(self.beta))
            print("=============================================")

            print("Num Most Probable States : " + str(count_non_zero))
            print("Entropy : " + str(self.get_entropy()))
            print("---------------------------------------------")

            if plot_dist:
                plot_bargraph_desc_order(
                    self.boltzmann_pd,
                    label="Boltzmann Dist.",
                    plot_first_few=count_non_zero,
                )
                plt.ylabel(ylabel)

        else:
            raise RuntimeError(
                "Please Run Exact Sampling at any specified temperature first"
            )

    def get_boltzmann_distribution(
        self,
        beta: float = 1.0,
        sorted: bool = False,
        save_distribution: bool = False,
        return_dist: bool = True,
        plot_dist: bool = False,
        verbose: bool = False,
    ) -> dict:
        """Get normalised boltzmann distribution over states

        ARGS:
        ----
        beta : inverse temperature (1/ T)
        sorted  : if True then the states are sorted in in descending order of their probability
        save_dist : if True then the boltzmann distribution is saved as an attribute of this class -> boltzmann_pd
        plot_dist : if True then plots histogram corresponding to the boltzmann distribution

        RETURNS:
        -------
        'dict' corresponding to the distribution
        """
        all_configs = [
            f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))
        ]
        bltzmann_probs = dict(
            [
                (state, self.get_boltzmann_factor(state, beta=beta))
                for state in tqdm(
                    all_configs,
                    desc="running over all possible configurations",
                    disable=not verbose,
                )
            ]
        )
        partition_sum = np.sum(np.array(list(bltzmann_probs.values())))
        prob_vals = list(
            np.array(list(bltzmann_probs.values())) * (1.0 / partition_sum)
        )

        bpd = dict(zip(all_configs, prob_vals))
        bpd_sorted_desc = value_sorted_dict(bpd, reverse=True)

        if save_distribution:
            self.boltzmann_pd = DiscreteProbabilityDistribution(bpd_sorted_desc)

        if plot_dist:
            plt.figure(2)
            plot_bargraph_desc_order(
                bpd_sorted_desc, label="analytical", plot_first_few=30
            )
            plt.legend()

        if return_dist:
            if sorted:
                return bpd_sorted_desc
            else:
                return bpd

    def run_exact_sampling(self, beta: float, verbose: bool = False) -> None:
        """Running this function executes the 'get_boltzmann_distribution' function, thus exhaustively enumerating all possible
        configurations of the system and saving the ditribution as an attribute 'boltzmann_pd'.

        NOTE:   This saves the requirement of recalculating the analytical distribution for any of the functions depending explicitly
                on the analytical boltzmann distribution.
                Run this function before calling any of the methods that uses the analytical boltzmann distribution.
                It is recommended not to run this for num_spins > 20, as it is highly ineffecient.

        ARGS:
        ----
        beta : inverse temperature

        """
        self.exact_sampling_status = True
        self.beta = beta
        if verbose:
            print("Running Exact Sampling | beta : ", beta)
        self.get_boltzmann_distribution(
            beta=beta, save_distribution=True, return_dist=False, verbose=verbose
        )
        if verbose:
            print("saving distribution to model ...")

    def get_observable_expectation(self, observable) -> float:
        """Return expectation value of a classical observables

        ARGS :
        ----
        observable: Must be a function of the spin configuration which takes an 'np.array' / 'str' of binary elements as input argument and returns a 'float'
        beta: inverse temperature

        """
        # all_configs = np.array(list(itertools.product([1, 0], repeat=self.num_spins)))
        all_configs = [
            f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))
        ]

        return sum(
            [self.boltzmann_pd[config] * observable(config) for config in all_configs]
        )

    def get_entropy(self):
        tmp = sorted(np.array(list(self.boltzmann_pd.values())), reverse=True)
        entropy = 0
        for val in tmp:
            if val > 0.00001:
                entropy += -1 * val * np.log2(val)
            else:
                return entropy

    def get_kldiv(self, q: dict, beta: Union[float, None] = None) -> float:
        """Return calculated KL-divergence of the boltzmann distribution wrt. a given distribution i.e
        D_kl( boltzmann|| q)

        ARGS:
        ----
        q : given distribution
        beta : inverse temperature of the model

        """
        ## check current beta and exact-sampling status
        if beta == None:
            beta = self.beta
        elif isinstance(beta, float):
            if beta != self.beta:
                raise ValueError(
                    "Current beta is different from model beta. Please 'run_exact_sampling' with appropriate beta value "
                )
                # bltz_dist = self.get_boltzmann_distribution(beta= beta)
        if self.exact_sampling_status:
            bltz_dist = self.boltzmann_pd
        else:
            bltz_dist = self.get_boltzmann_distribution(beta=beta)

        ## check q
        q_vals = list(q.values())
        assert np.sum(q_vals) == 1, " given distribution is not normalised "
        all_configs = [
            f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))
        ]
        if set(q.keys()) != set(all_configs):
            raise ValueError(
                " given distribution is not defined over all possible configurations "
            )

        ## re-order
        bltz_dist = DiscreteProbabilityDistribution(bltz_dist)
        q = DiscreteProbabilityDistribution(q)
        q.normalise()

        ## calc
        bltz_dist = bltz_dist.index_sorted_dict()
        q = q.index_sorted_dict()

        p = list(bltz_dist.values())
        q = list(q.values())

        return sum(p[i] * np.log2(p[i] / q[i]) for i in range(len(p)) if p[i] != 0)

    def get_jsdiv(self, q, beta: Union[float, None] = None) -> float:
        """Return calculated KL-divergence of the boltzmann distribution wrt. a given distribution i.e
        D_js( boltzmann ,  q)

        ARGS:
        ----
        q : given distribution
        beta : inverse temperature of the model

        """
        ## check current beta and exact-sampling status
        if beta == None:
            beta = self.beta
        elif isinstance(beta, float):
            if beta != self.beta:
                raise ValueError(
                    "Current beta is different from model beta. Please 'run_exact_sampling' with appropriate beta value "
                )
                # bltz_dist = self.get_boltzmann_distribution(beta= beta)
        if self.exact_sampling_status:
            bltz_dist = self.boltzmann_pd
        else:
            bltz_dist = self.get_boltzmann_distribution(beta=beta)

        ## checks
        q_vals = list(q.values())
        assert np.sum(q_vals) == 1, " given distribution is not normalised "
        all_configs = [
            f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))
        ]
        assert set(q.keys()).issubset(
            all_configs
        ), " given distribution is not defined over all possible configurations "

        ## create mixed distribution
        m = {}
        for key in bltz_dist.keys():
            m[key] = 0.5 * (bltz_dist[key] + q[key])

        return 0.5 * self.get_kldiv(bltz_dist, m) + 0.5 * self.get_kldiv(q, m)


def random_ising_model(
    n_spins: int,
    seed: int,
    print_model: bool = False,
):

    np.random.seed(seed)

    ## construct problem Hamiltonian ##
    shape_of_J = (n_spins, n_spins)

    ## defining J matrix (mutual 1-1 interaction)
    # J =  np.round(np.random.choice([+1, 0, -1], size=(n_spins, n_spins)), decimals=2)
    J = np.random.uniform(low=-1, high=1, size=shape_of_J)

    J = 0.5 * (J + J.transpose())
    J = np.round(J - np.diag(np.diag(J)), decimals=3)

    # defining h
    h = np.round(0.4 * np.random.randn(n_spins), decimals=2)
    # h = np.round(np.random.uniform(low= -1, high = 1, size= (n_spins)), decimals=2)

    param_model = IsingEnergyFunction(J, h, name="param_model")

    if print_model:
        param_model.model_summary()

    return param_model
