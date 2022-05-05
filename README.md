# FPM_diffusion
Boilerplate code for simulating FPMs, computing quantities from the diffusion approximations, and comparing diffusion with simulation.

File contents:
* `how_to.ipynb` -- notebook with a tutorial;
* `diffusion.py` -- functions for computing quantities from the diffusion approximation given the drift function `a(x)` and the diffusion function `b(x)`;
* `simulation.py` -- various functions to simulate Wright-Fisher-type models with two alleles given the probability of sampling a given allele, encapsulated in the iteration function;
* `comparison_plots.py` -- code to compare results obtained via the simulation and the diffusion approximation;
