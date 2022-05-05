import numpy as np
from tqdm import tqdm
from math import floor

from diffusion import *

""" EXAMPLE ITERATION FUNCTIONS """

def iter_neutral_WF():
    """Example iteration function for the neutral WF mdel.

    The function produces a function np.array -> np.array of abitrary length and shape (1,)
    that gives the offspring generation of the population from the parental one.
    Two alleles/variants, A and B, are encoded by 0 and 1 respectively.

    Returns:
        func: function np.array -> np.array described above.
    """
    def ans(population):
        x = population.mean()
        return np.random.binomial(1,x,len(population))
    return ans 


def iter_conformity_3rm(s, D):
    """Example iteration function for the FPM of conformity with n=3 role models.

    The function produces a function np.array -> np.array of abitrary length and shape (1,),
    that gives the offspring generation of the population from the parental one.
    Two alleles/variants, A and B, are encoded by 0 and 1 respectively.

    Args:
        s (float): selection coefficient
        D (float): conformity coefficient

    Returns:
        func: function np.array -> np.array described above.
    """
    def ans(population):
        x = population.mean()
        P = ((1+s)*(x + D*x*(1-x)*(2*x-1))/(1+s*(x + D*x*(1-x)*(2*x-1))))
        return np.random.binomial(1,P,len(population))
    return ans

def iter_conformity_5rm(s, D3, D4): 
    """Example iteration function for the FPM of conformity with n=5 role models.

    The function produces a function np.array -> np.array of abitrary length and shape (1,),
    that gives the offspring generation of the population from the parental one.
    Two alleles/variants, A and B, are encoded by 0 and 1 respectively.

    Args:
        s (float): selection coefficient.
        D3 (float): conformity coefficient D(3).
        D4 (float): conformity coefficient D(4).

    Returns:
        func: function np.array -> np.array described above.
    """
    def ans(population):
        x = population.mean()
        P = (1 + s)*(x + (1 -x)*x*(2*x - 1)*(D4- x*(1 - x)*(D4 - 2*D3)))/(1 + s*(x + (1 -x)*x*(2*x - 1)*(D4- x*(1 - x)*(D4 - 2*D3))))
        return np.random.binomial(1,P,len(population))
    return ans

""" FPM SIMULATION FUNCTIONS """

def simulate(iter_fn, n_range, N, p_init=0.5, seed=42):
    """Basic simulation function.

    Simulate a single run of fixed length (number of generations)

    Args:
        iter_fn (func): iteration function, described above.
        n_range (int): range of the simulation, number of generations to simulate.
        N (int): population size, the actual size of the list encoding the individuals in the population
        p_init (float, optional): initial frequency. Defaults to 0.5.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        list(float): list containing frequencies of allele "A" in each generation.
    """
    np.random.seed(seed)
    population = np.array([1]*floor(N*p_init) + [0]*(N-floor(N*p_init)))
    assert N == len(population)

    out = [population.mean()]
    for i in range(n_range+1):
        population = iter_fn(population)
        out.append(population.mean())

    return out

def simulate_rep(iter_fn, n_rep, n_range, N, p_init=0.5, seed=42):
    """Basic simulation function repeated a given number of times.

    Simulate a n_rep runs of fixed length (number of generations)

    Args:
        iter_fn (func): iteration function, described above.
        n_rep (int): number of independent simulation runs.
        n_range (int): range of the simulation, number of generations to simulate.
        N (int): population size, the actual size of the list encoding the individuals in the population
        p_init (float, optional): initial frequency. Defaults to 0.5.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        np.array(float): array of share (n_range, n_rep) 
                         containing frequencies of allele "A" in generation in each run.
    """
    np.random.seed(seed)
    return np.array([simulate(iter_fn, n_range, N, p_init) for k in range(n_rep)]).T

def simulate_until_absorption(iter_fn, n_rep, N, p_init=0.5, generation_cap=10e8, seed=123):
    """Simulation until absorption.

    Simulate a n_rep runs of arbitrary length, as much as it takes for the population to reach absorption.

    Args:
        iter_fn (func): iteration function, described above.
        n_rep (int): number of independent simulation runs.
        N (int): population size, the actual size of the list encoding the individuals in the population
        p_init (float, optional): initial frequency. Defaults to 0.5.
        generation_cap (int, optional): upper limit on the number of generations to simulate 
                                        (to save computational resources). Defaults to 10e8.
        seed (int, optional): random seed. Defaults to 123.

    Returns:
        freqs (np.array(float)): array of shape (n_rep,) containing number of generations it took for
                                 the population to reach an absorption state.
        P_A (float): fraction of populations that fixed to p=1.
    """
    out = []
    P_A = []
    for _ in range(n_rep):
        gen = 0
        A = 0

        population = np.array([1]*round(N*p_init) + [0]*(N-round(N*p_init)))
        freq = population.mean()

        while freq != 0.0 and freq != 1.0:
            population = iter_fn(population)
            freq = population.mean()
            gen += 1
            if gen > generation_cap:
                break
        
        if gen > generation_cap:
                break

        A = 1 if freq == 1.0 else 0

        out.append(gen)
        P_A.append(A)

    return np.array(out), np.mean(P_A)
