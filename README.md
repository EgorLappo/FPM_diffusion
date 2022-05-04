# FPM_diffusion
Boilerplate code for simulating FPMs, computing functions from diffusion approximations, and comparing it

File contents:
* `diffusion.py` -- functions for computing quantities from the diffusion approximation given the drift function `a(x)` and the diffusion function `b(x)`;
* `simulation.py` -- various functions to simulate Wright-Fisher-type models with two alleles given the probability of sampling a given allele, encapsulated in the iteration function.
* `comparison_plots.py` -- code to compare results obtained via the simulation and the diffusion approximation.