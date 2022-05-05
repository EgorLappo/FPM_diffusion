import numpy as np
from simulation import *
from diffusion import *
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm


def diffusion_simulation_compare(iter_fn, a, b, N, n_rep=250, n_points=50, labels=['A','B'], use_tqdm = True, suptitle=''):
    """Plot comparison of simulation and diffusion approximation of the FPM.

    Plots values of P_1(p) and t_bar(p) computed empirically and via diffusion against each other.

    Args:
        iter_fn (func): iteration function for the FPM.
        a (func): drift function.
        b (func): diffusion funcion.
        N (int): population size.
        n_rep (int, optional): number of populations to simulate for each initial frequency. Defaults to 250.
        n_points (int, optional): number of subdivisions of [0,1] used as initial frequencies. Defaults to 50.
        labels (list, optional): subplot labels, used in case one wishes to create a stacked figure. Defaults to ['A','B'].
        use_tqdm (bool, optional): whether to monitor progress with tqdm. Defaults to True.
        suptitle (str, optional): figure subtitle. Defaults to ''.

    Returns:
        matplotlib.Figure: resulting figure. To save it one only needs to run fig.savefig(filename).
    """
    xs = np.linspace(0.0, 1.0, n_points)

    if tqdm: 
        sim = [simulate_until_absorption(iter_fn, n_rep, N, p_init=x) for x in tqdm(xs)]
    else:
        sim = [simulate_until_absorption(iter_fn, n_rep, N, p_init=x) for x in xs]

    print('finished the FPM simulations!')

    if use_tqdm:
        P_1_ys_theory = np.array([P1(x, a, b) for x in tqdm(xs)])
    else:
        P_1_ys_theory = np.array([P1(x, a, b) for x in xs])
    P_1_ys_sim = [x[1] for x in sim]

    if use_tqdm:
        t_bar_ys_theory = N*np.array([t_bar(x, a, b) for x in tqdm(xs)])
    else:
        t_bar_ys_theory = N*np.array([t_bar(x, a, b) for x in xs])
    t_bar_ys_sim = np.array(sum([[(x, y) for y in z[0]] for x, z in zip(xs, sim)],[]))

    print('done computing diffusion values!')

    plt.style.use('seaborn-paper')

    fig, axs = plt.subplots(1,2, figsize=(8,3), constrained_layout=True)

    sns.lineplot(x=xs, y=P_1_ys_theory, label="Diffusion", ax=axs[0])
    sns.lineplot(x=xs,y=P_1_ys_sim, label="Simulation", ax=axs[0])
    axs[0].set_aspect(1.0)

    axs[0].set_xlabel('p')
    axs[0].set_ylabel('$P_1(p)$')
    axs[0].set_title('Probability of fixation')
    axs[0].get_legend().remove()

    axs[0].annotate(labels[0], (-0.15,1.05), xycoords='axes fraction', fontsize=12, fontweight="bold")

    sns.lineplot(x=xs, y=t_bar_ys_theory, label="Diffusion", ax=axs[1])
    sns.lineplot(x=t_bar_ys_sim[:,0],y=t_bar_ys_sim[:,1], label="Simulation", ax=axs[1], ci='sd')
    axs[1].set_xlabel('p')
    axs[1].set_ylabel('$\\bar{t}(p)$')
    axs[1].set_title('Time to absorption')

    axs[1].annotate(labels[1], (-0.17,1.05), xycoords='axes fraction', fontsize=12, fontweight="bold")

    fig.suptitle(suptitle)

    return fig