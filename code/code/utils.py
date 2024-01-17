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
import copy
import matplotlib.pyplot as plt
from amm_cython import amm_cython
from amm import amm
from params import params
import logging
import datetime
import subprocess
import os

def get_current_git_branch():
    try:
        output = subprocess.check_output(["git", "branch"], text=True)
        branch_name = output.strip()
        logging.info(f"Current branch:\n{branch_name}\n")
        return branch_name
    except subprocess.CalledProcessError as e:
        logging.info(f"Error: {e}")
        return None

def logging_config(filename):
    if os.getenv("PBS_JOBID") != None:
        job_id = os.getenv("PBS_JOBID")
    else:
        job_id = os.getpid()
    logging.basicConfig(filename=f'output/{filename}_{job_id}.log', format='%(message)s', level=logging.INFO)
    # Log the time
    logging.info(f"Time: {datetime.datetime.now()}\n")
    return None

def calculate_cvar(log_returns):
    """
    Calculate the CVaR of a set of returns.
    """
    global params
    var = np.quantile(-log_returns, params['alpha'])
    cvar = np.mean(-log_returns[-log_returns >= var])
    return cvar, var

def cvar_unconstrained(cvar, initial_pools_dist, lambda1):
    lambda2, lambda3, lambda3 = 10*np.ones(3)
    global penalties
    penalties = lambda1 * (np.sum(initial_pools_dist) - 1)**2 + lambda2 * max(0, params['q'] - probability) + lambda3 * np.sum(np.maximum(-initial_pools_dist, 0))
    return cvar + penalties

def calculate_log_returns(x0, final_pools_dists, l):
    """
    In the simulate method of the amm class, a  certain number of trading paths are
    generated. Consequently, for each path we have a final log return.
    This function calculates the log returns for each final state of the pools over
    all the simulated paths.
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

def portfolio_evolution(initial_pools_dist, random_numbers, params, unconstrained=False):

    # Initialize the pools
    amm_instance = amm_cython(params['Rx0'], params['Ry0'], params['phi'])
    # Check if there is a negative weight
    if np.any(initial_pools_dist < 0):
        logging.info(f'Negative weight: {initial_pools_dist}')
        return 1e6

    # Compute the actual tokens for each pool. The initial_pools_dist are the
    # weights of the pools. We need to multiply them by the initial wealth
    X0 = params['x_0'] * initial_pools_dist

    # Evaluate the number of LP tokens. This will be used to compute the returns
    try:
        l = amm_instance.swap_and_mint(X0)
    except AssertionError as e:
        logging.info(f"Error: {e}")
        return 1e6

    # Simulate the evolution of the pools (scenario simulation). We simulate params['batch_size'] paths, 
    # hence we will have params['batch_size'] amount of returns at the end.
    np.random.seed(params['seed'])

    final_pools_dists, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = amm_instance.simulate(
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

    if unconstrained == True:
        global lambda1
        cvar_pen = cvar_unconstrained(cvar, initial_pools_dist, lambda1)
        return cvar_pen
    else:
        return cvar

def constraint_1(x):
    return np.sum(x) - 1

def constraint_2(x):
    global probability
    return probability - params['q']

def optimize_distribution(params, method, unconstraint=False):
    """
    Optimizes the distribution of wealth across liquidity pools to minimize CVaR,
    conditioned to P[final return > 0.05]>params['q'].

    Args:
    - amm_instance (amm): Instance of the amm class.
    - params (dict): Parameters for the amm and optimization.

    Returns:
    - dict: Optimal weights and corresponding CVaR.
    """
    np.random.seed(params['seed'])
    # Global variables to store the log returns and the
    # probability of having a return greater than 0.05
    global probability
    global log_returns
    log_returns, probability = 0, 0

    # Constraints, bounds and options for the optimization
    constraints = [{'type': 'eq', 'fun': constraint_1},
                {'type': 'ineq', 'fun': constraint_2}]
    bounds_initial_dist = [(1e-5, 1) for i in range(params['N_pools'])]
    options = {'maxiter': 1000, 'ftol': 1e-8}

    # Instantiate the amm class
    amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])

    # Callback function to print the current CVaR and the current parameters
    def callback_function(x, *args):
        logging.info(f"Current initial_dist: {x} -> Sum: {np.sum(x)}")
        logging.info(f"Current probability: {probability}")
        try:
            logging.info(f'Penalities: {penalties}')
        except:
            pass
        logging.info(f'Mean loss: {np.mean(-log_returns)}')
        logging.info(f"Current VaR:{np.quantile(-log_returns, params['alpha'])}")
        logging.info(f"Current CVaR: {cvar}\n")
    
    def callback_lambda_update(x, *args):
        global lambda1
        if np.abs(np.sum(x) - 1) < 1e-4:
            lambda1 = lambda1 * 1.5
        logging.info(f"lambda1: {lambda1}")
    
    def wrapper_callback(x, *args):
        callback_function(x, *args)
        callback_lambda_update(x, *args)

    # The following while loop is used to check if the initial distribution of wealth
    # across pools is feasible. If it is not, a new one is generated.
    cond = True
    while cond == True:
        # Initial distribution of wealth across pools
        random_vector = np.random.uniform(0, 100, params['N_pools'])
        initial_guess = random_vector / np.sum(random_vector)
        try:
            amm_instance.swap_and_mint(initial_guess*params['x_0'], quote=True)
            cond = False
        except ValueError as e:
            logging.info(f"Error: {e}")
    logging.info(f"Initial guess:\n\t{initial_guess}\n")

    # Optimization procedure
    logging.info("Starting...")
    if unconstraint == False:
        logging.info(f"Minimization of vanilla cVaR")
        logging.info(f"Optimization method: {method}")
        result = minimize(portfolio_evolution, initial_guess, args=(amm_instance, params),
                method=method, bounds=bounds_initial_dist, constraints=constraints, options=options, callback=callback_function)
    else:
        logging.info(f"Unconstrained minimization of vanilla cVaR")
        logging.info(f"Optimization method: {method}")
        global lambda1
        lambda1 = 10
        result = minimize(portfolio_evolution, initial_guess, args=(amm_instance, params, True),
                    method=method, callback=wrapper_callback)

    logging.info(f"Results:\n\t{result}")

    return result.x


def simulation_plots(res, random_numbers, params):
    """
    Plots the evolution of the reserves, the price and the returns for a given
    initial distribution of wealth across pools.
    """

    X0 = res[-params['N_pools']:] * params['x_0']

    # Initialize the pools
    amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])

    # Evaluate the number of LP tokens
    l = amm_instance.swap_and_mint(X0)

    # Simulate the evolution of the pools.
    # final_pools_dist is a list of length batch_size. Each element of the list contains 
    # the final reserves of the pools for a given path. To access the final X reserve of
    # the i-th path you need to do final_pools_dist[i].Rx. Same for Ry.
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
    plt.savefig(f'pools_{os.getenv("PBS_JOBID")}.png')

    # Plot the distribution of the returns
    plt.figure(figsize=(10, 8), tight_layout=True)
    plt.hist(log_returns, bins=50, alpha=0.7)
    plt.axvline(-cvar, color='r', linestyle='dashed', linewidth=1, label='CVaR')
    plt.axvline(-var, color='b', linestyle='dashed', linewidth=1, label='VaR')
    plt.axvline(0.05, color='g', linestyle='dashed', linewidth=1, label=r'$\xi$', alpha=0.0)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend([f'CVaR:{-cvar}', f'VaR:{-var}', fr'$E[r_T]$:{np.mean(log_returns):.3f}', fr'$P[r_T>\xi]$:{probability:.3f}'])
    plt.savefig(f'returns_{os.getenv("PBS_JOBID")}.png')

    plt.show()

def worker(rgs_seed_start, rgs_seed_end, params, random_numbers, result_queue):
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

class KernelRidge_Warper():
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