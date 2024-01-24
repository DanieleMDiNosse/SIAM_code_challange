import numpy as np
from scipy.optimize import minimize
import copy
import matplotlib.pyplot as plt
from amm_cython import amm_cython
from amm import amm
from params import params
import logging
import os

class KernelRidge_Warper():
    '''Warper class for the KernelRidge class of sklearn.'''
    def __init__(self, options):
        from sklearn.kernel_ridge import KernelRidge
        self.krr = KernelRidge(**options)

    def fit(self, x, y):
        self.krr.fit(x, y)

    def score(self, x, y):
        return self.krr.score(x, y)
    
    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.krr.predict(x)

def calculate_cvar(log_returns):
    """
    Calculate the CVaR of a set of returns.

    Parameters
    ----------
    log_returns : array
        Distribution of log returns.
    
    Returns
    -------
    cvar : float
        CVaR at level params['alpha'].
    """
    global params
    var = np.quantile(-log_returns, params['alpha'])
    cvar = np.mean(-log_returns[-log_returns >= var])
    return cvar, var

def calculate_log_returns(x0, final_pools_dists, l):
    """
    In the simulate method of the amm class, a certain number of trading paths are
    generated. Consequently, for each path we have a final log return.
    This function calculates the distribution of log returns for each final state of 
    the pools over all the simulated paths.

    Parameters
    ----------
    x0 : array
        Initial wealth distribution across pools.
    final_pools_dists : list
        List of wealth final distributions across pools for each path.
    l : float
        Number of LP tokens.
    
    Returns
    -------
    log_returns : array
        Distribution of log returns.
    """
    x_T = np.zeros(params['batch_size']) # each element of x_T will be the final wealth of a path

    # In order to have the final wealth, we need to burn and swap the LP tokens
    # for each path. This is done by the burn_and_swap method of the amm class.
    # The method takes all the LP tokens, burn them and swap coin-Y for coin-X.
    for k in range(params['batch_size']):
        x_T[k] = np.sum(final_pools_dists[k].burn_and_swap(l))

    # Calculate the initial wealth
    x_0 = np.sum(x0)

    # Calculate the log returns for each path
    log_returns = np.log(x_T) - np.log(x_0)

    return log_returns

def portfolio_evolution(initial_pools_dist, random_numbers, params):
    '''
    Simulate the evolution of the pools and calculate the CVaR of the returns.
    
    Parameters
    ----------
    initial_pools_dist : array
        Initial weight vector.
    random_numbers : dict
        Dictionary of random numbers.
    params : dict
        Dictionary of parameters.

    Returns
    -------
    cvar : float
        CVaR of the returns.
    '''

    # Initialize the pools
    amm_instance = amm_cython(params['Rx0'], params['Ry0'], params['phi'])

    # Check if there is a negative weight
    if np.any(initial_pools_dist < 0):
        logging.info(f'Negative weight: {initial_pools_dist}')
        return 1e6

    # Compute the actual wealth distribution across the pools.
    X0 = params['x_0'] * initial_pools_dist

    # Evaluate the number of LP tokens. This will be used to compute the returns.
    try:
        l = amm_instance.swap_and_mint(X0)
    except AssertionError as e:
        logging.info(f"Error: {e}")
        return 1e6

    # Simulate the evolution of the pools (scenario simulation).
    np.random.seed(params['seed'])

    final_pools_dists, _, _, _, _, _ = amm_instance.simulate(
            kappa=np.array(params['kappa']),
            p=np.array(params['p']),
            sigma=np.array(params['sigma']),
            T=params['T'],
            N_list=random_numbers['N_list'],
            event_type_list=random_numbers['event_type_list'],
            event_direction_list=random_numbers['event_direction_list'],
            v_list=random_numbers['v_random_number_list'],
            batch_size=params['batch_size'])

    # Calculate the log returns for each path
    global log_returns
    log_returns = calculate_log_returns(X0, final_pools_dists, l)

    # Compute the cvar
    global cvar
    cvar, _ = calculate_cvar(log_returns)

    # Compute the probability of having a return greater than 0.05
    global probability
    probability = log_returns[log_returns > 0.05].shape[0] / log_returns.shape[0]

    return cvar

def constraint_1(x):
    return np.sum(x) - 1

def constraint_2(x):
    global probability
    return probability - params['q']

def simulation_plots(res, random_numbers, params):
    """
    Plot the evolution of the reserves, the price and the log returns for a given
    initial distribution of wealth across pools.

    Parameters
    ----------
    res : array
        Optimal initial weight vector.
    random_numbers : dict
        Dictionary of random numbers.
    params : dict
        Dictionary of parameters.
    
    Returns
    -------
    None
    """

    X0 = res[-params['N_pools']:] * params['x_0']

    # Initialize the pools
    amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])

    # Evaluate the number of LP tokens
    l = amm_instance.swap_and_mint(X0)

    # Simulate the evolution of the pools.
    np.random.seed(params['seed'])

    XT, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = amm_instance.simulate_fast(kappa=np.array(params['kappa']),
            p=np.array(params['p']),
            sigma=np.array(params['sigma']),
            T=params['T'],
            N_list=random_numbers['N_list'],
            event_type_list=random_numbers['event_type_list'],
            event_direction_list=random_numbers['event_direction_list'],
            v_list=random_numbers['v_random_number_list'],
            batch_size=params['batch_size'])

    # Calculate the log returns, cvar and var
    log_returns = calculate_log_returns(X0, XT, l)
    probability = log_returns[log_returns > 0.05].shape[0] / log_returns.shape[0]
    cvar, var = calculate_cvar(log_returns)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
    i = np.random.randint(0, params['N_pools'])
    # Plot the evolution of the reserves
    ax[0].plot(Rx_t[i])
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('X-Reserves')

    ax[1].plot(Ry_t[i])
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Y-Reserves')

    # Plot the evolution of the marginal price
    ax[2].plot(np.array(Rx_t[i])/np.array(Ry_t[i]))
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Marginal Price')
    plt.savefig(f'output/pools.png')

    # Plot the distribution of the returns
    plt.figure(figsize=(10, 8), tight_layout=True)
    plt.hist(log_returns, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(-cvar, color='r', linestyle='dashed', linewidth=1, label='CVaR')
    plt.axvline(-var, color='b', linestyle='dashed', linewidth=1, label='VaR')
    plt.axvline(0.05, color='g', linestyle='dashed', linewidth=1, label=r'$\xi$', alpha=0.0)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend([f'CVaR:{-cvar}', f'VaR:{-var}', fr'$E[r_T]$:{np.mean(log_returns):.3f}', fr'$P[r_T>\xi]$:{probability:.3f}'])
    plt.savefig(f'output/returns.png')

    plt.show()

    return None

def worker(rgs_seed_start, rgs_seed_end, params, random_numbers, result_queue):
    '''
    Worker function for the multiprocessing module. Each process will call this
    function to simulate a certain number of paths.
    
    Parameters
    ----------
    rgs_seed_start : int
        Start of the range of random seeds.
    rgs_seed_end : int
        End of the range of random seeds.
    params : dict
        Dictionary of parameters.
    random_numbers : dict
        Dictionary of previously generated random numbers.
    result_queue : multiprocessing.Queue
        Queue to collect the results.
    
    Returns
    -------
    None
    '''
    local_x_data, local_y_data = [], []
    for rgs_seed in range(rgs_seed_start, rgs_seed_end):
        np.random.seed(rgs_seed)
        theta = []
        for _ in range(len(params['Rx0'])):
            theta.append(np.random.uniform())
        theta = np.array(theta) / np.sum(theta)

        cvar = portfolio_evolution(theta, random_numbers, params)

        local_x_data.append(theta)
        local_y_data.append(cvar)

    result_queue.put((local_x_data, local_y_data))

    return None