# Fixed-point problems in Wasserstein space

Various statistical tasks, such as  sampling or computing Wasserstein barycenters, can be reformulated as fixed-point problems for operators on probability distributions. 
The goal is to accelerate the fixed-point iterations with Riemannian Anderson mixing. 

## Installation
```
pip install --upgrade fpw@git+https://github.com/viviaxenov/fpw
```
### Requirements
  - `cvxpy` for $l_\infty$ regularized minimization
  - `torch`
### Optional requirements
  - `emcee` for sampling from general distributions
  - `pymanopt` for comparison with Riemannian minimization methods
  - `tqdm` for cool progress bar in the testing script 

## Examples
### Gaussian case
One, two, three
### General case
TBD
### Documentation
## Citation
TBD
