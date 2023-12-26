'''This script contains functions that are used for the task 2 of the SIAM competition.
The task required to find the distribution of initial wealth across n pools such that
it minimize the conditional Value at Risk (CVaR) of the final wealth distribution subject
to the constraint that the probability of having a return of more than 0.05 is greater
than 0.7.

Conceptually, the procedure is as follows:
1. Choose an initial wealth distribution
2. Simulate the evolution of the wealth distribution over time using the simulate
module of the amm class. This module generates a certain number N of paths that are
characterized by random events that result in a certain final wealth distribution.
3. Calculate the CVaR of the final wealth distribution
4. Check the constraint.
'''

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from amm import amm
from params import params

def calculate_cvar(log_returns, alpha=0.95):
    """
    Calculate the CVaR of a set of returns.
    """

    quantile = np.quantile(-log_returns, alpha)
    cvar = np.mean(-log_returns[-log_returns >= quantile])
    return cvar

def calculate_log_returns(initial_pools_dist, final_pools_dists, l):
    """
    In the simulate method of the amm class, a  certain number of trading paths are
    generated. Consequently, for each path we have a final log return.
    This function calculates the log returns for each final state of the pools over
    all the simulated paths.
    """

    batch_size = len(final_pools_dists) # number of simulated paths
    x_T = np.zeros(batch_size) # each element of x_T is the final wealth of a path

    # In order to have the final wealth, we need to burn and swap the LP tokens
    # for each path. This is done by the burn_and_swap method of the amm class.
    for k in range(batch_size):
        x_T[k] = np.sum(final_pools_dists[k].burn_and_swap(l))

    x_0 = np.sum(initial_pools_dist)
    log_returns = np.log(x_T) - np.log(x_0)
    return log_returns#, mean_log_return, std_log_return

def objective_function(initial_pools_dist, params):
    """
    Objective function to minimize. It takes as input the initial wealth distribution
    and the parameters of the model, togerher with the amm class instance. 
    It returns the CVaR of the final return distribution.
    """

    # Extract the parameters from the dictionary params
    N = params['N_pools']
    phi = params['phi']
    kappa, p, sigma = params['kappa'], params['p'], params['sigma']
    T = params['T']
    batch_size = params['batch_size']

    # Initialize the pools
    Rx0 = initial_pools_dist[:N]
    Ry0 = initial_pools_dist[N:]
    amm_instance = amm(Rx0, Ry0, phi)
    # Evaluate the number of LP tokens
    l = amm_instance.swap_and_mint(initial_pools_dist)

    # Simulate the evolution of the pools.
    # final_pools_dist is a list of length batch_size. Each element of the list contains 
    # the final reserves of the pools for a given path. To access the final X reserve of
    # the i-th path you need to do final_pools_dist[i].Rx. Same for Ry.
    final_pools_dists, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = amm_instance.simulate(kappa=kappa, p=p, sigma=sigma, T=T, batch_size=batch_size)
    log_returns = calculate_log_returns(initial_pools_dist, final_pools_dists, l)

    # Check if return exceeds 0.05
    # this pieace of code can be improved using masks
    count_success = 0
    for log_return in log_returns:
        if log_return > 0.05:
            count_success += 1

    global probability
    probability = count_success / batch_size

    cvar = calculate_cvar(log_returns)
    return cvar

def constraint_1(x):
    return np.sum(x) - 1

def constraint_2(x):
    return x

def constraint_3(x):
    global probability
    return probability - 0.7

def optimize_distribution(amm_instance, params):
    """
    Optimizes the distribution of wealth across liquidity pools to minimize CVaR.

    Args:
    - amm_instance (amm): Instance of the amm class.
    - params (dict): Parameters for the amm and optimization.

    Returns:
    - dict: Optimal weights and corresponding CVaR.
    """
    # Constraints for optimization
    constraints = [{'type': 'eq', 'fun': constraint_1},
                   {'type': 'ineq', 'fun': constraint_2},
                   {'type': 'ineq', 'fun': constraint_3}]

    # Initial guess (even distribution across pools)
    initial_pools_dist = np.ones(params['N_pools'])*params['x_0']

    # Optimization
    result = minimize(objective_function, initial_pools_dist, args=(params),
                      method='SLSQP', constraints=constraints)

    # Return the optimal weights and corresponding CVaR
    optimal_weights = result.x
    optimal_cvar = objective_function(optimal_weights, amm_instance, params)

    return {'weights': optimal_weights, 'cvar': optimal_cvar}

