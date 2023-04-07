# **quMCMC**
This repository contains the python implementation of Quantum Enhanced Markov Chain Monte Carlo. The code is based mainly on the algorithm reported by David Layden et al. in **Quantum-enhanced Markov chain Monte Carlo**. The code provides support for reproducing most of the results discussed in the original paper, and the adavantages of the Quantum Enhanced MCMC (*quMCMC*) over classical MCMC has been highlighted.




### **Installation**

The package can be installed using `pip` in the main directory of the package as,

```bash
>>> pip install .
```
### **Example Usage**
```bash
## DEFINE MODEL
n_spins = 10

## construct problem Hamiltonian ##
shape_of_J=(n_spins,n_spins)

## defining J matrix (mutual 1-1 interaction)
J =  np.random.uniform(low= -2, high= 2, size= shape_of_J )
J = 0.5 * (J + J.transpose() )

J = np.round( J - np.diag(np.diag(J)) , decimals= 3)

# defining h
h = np.round(0.5 * np.random.randn(n_spins), decimals=2)

# instantiate the model
model = IsingEnergyFunction(J, h, name= 'my_model')
```

```bash
=============================================
            MODEL : my_model
=============================================
Non-zero Interactions (J) : 45 / 45
Non-zero Bias (h) : 9 / 10
---------------------------------------------
Average Interaction Strength <|J|> :  0.5966799999999999
Average Bias Strength <|h|>:  0.5010000000000001
alpha :  0.5606804251097042
---------------------------------------------
```
```bash
## set current beta
beta = 1.100209

## RUN EXACT SAMPLING
exact_sampled_model = Exact_Sampling(model, beta)

steps = 10000
## RUN CLSSICAL SAMPLING
cl_chain =classical_mcmc(
    n_hops=steps,
    model=model,
    temperature=1/beta,
)
## RUN QUUANTUM SAMPLING
qamcmc_chain =quantum_enhanced_mcmc(
    n_hops=steps,
    model=model,
    temperature=1/beta,
)

```

For a more detailed example check the `tutorial.ipynb` 



### **References** 
1.  [**Quantum-enhanced Markov chain Monte Carlo**](https://www.arxiv-vanity.com/papers/2203.12497/) by David Layden et al.
2. [**Qulacs Simulator**](http://docs.qulacs.org/en/latest/intro/0_about.html)
3. [**Qiskit Aer Simulator**](https://qiskit.org/documentation/stubs/qiskit_aer.AerSimulator.html)

### **License**
The package is licensed under  ` MIT License`