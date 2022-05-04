import numpy as np
from tqdm import tqdm
from math import floor

from diffusion import *


def iter_3rm(s, D):
    def ans(population):
        x = population.mean()
        P = ((1+s)*(x + D*x*(1-x)*(2*x-1))/(1+s*(x + D*x*(1-x)*(2*x-1))))
        return np.random.binomial(1,P,len(population))
    return ans

def iter_5rm(s, D3, D4): 
    def ans(population):
        x = population.mean()
        P = (1 + s)*(x + (1 -x)*x*(2*x - 1)*(D4- x*(1 - x)*(D4 - 2*D3)))/(1 + s*(x + (1 -x)*x*(2*x - 1)*(D4- x*(1 - x)*(D4 - 2*D3))))
        return np.random.binomial(1,P,len(population))
    return ans

def simulate(iter_fn, n_range, N, p_init=0.5):

    population = np.array([1]*floor(N*p_init) + [0]*(N-floor(N*p_init)))
    assert N == len(population)

    out = [population.mean()]
    for i in range(n_range+1):
        population = iter_fn(population)
        out.append(population.mean())

    return out

def simulate_rep(iter_fn, n_rep, n_range, N, p_init=0.5, seed=42):
    np.random.seed(seed)
    return np.array([simulate(iter_fn, n_range, N, p_init) for k in range(n_rep)]).T

def simulate_until_absorption(iter_fn, n_rep, N, p_init=0.5, seed=123):
    out = []
    P_A = []
    for i in range(n_rep):
        gen = 0
        A = 0

        population = np.array([1]*round(N*p_init) + [0]*(N-round(N*p_init)))
        freq = population.mean()

        while freq != 0.0 and freq != 1.0:
            population = iter_fn(population)
            freq = population.mean()
            gen += 1


        A = 1 if freq == 1.0 else 0

        out.append(gen)
        P_A.append(A)

    return np.array(out), np.mean(P_A)

def simulate_trajectories(iter_fn, N, n_rep=36, n_range=None, p_init=0.5):
    if n_range==None:
        n_range = 5*N

    result = []

    for i in tqdm(range(n_rep)):
        freqs = []

        population = np.array([1]*round(N*p_init) + [0]*(N-round(N*p_init)))
        freq = population.mean()

        for j in range(n_range):
            population = iter_fn(population)
            freq = population.mean()
            freqs.append(freq)

        result += [freqs]

    result = np.array(result).T

    return result