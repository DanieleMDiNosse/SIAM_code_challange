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
import logging
import os

if os.getenv("PBS_JOBID") != None:
    job_id = os.getenv("PBS_JOBID")
else:
    job_id = os.getpid()

logging.basicConfig(filename=f'output_{job_id}.log', format='%(message)s', level=logging.INFO)

def calculate_cvar(log_returns, alpha=0.95):
    """
    Calculate the CVaR of a set of returns.
    """

    quantile = np.quantile(-log_returns, alpha)
    cvar = np.mean(-log_returns[-log_returns >= quantile])
    return cvar

def calculate_cvar_RU(gamma, u, alpha):
    """
    Calculate the CVaR of a set of returns following Rockafellar and Uryasev.
    """

    cvar_RU = gamma + 1 / (1 - alpha) * np.mean(u)
    return cvar_RU

def calculate_log_returns(x0, final_pools_dists, l):
    """
    In the simulate method of the amm class, a  certain number of trading paths are
    generated. Consequently, for each path we have a final log return.
    This function calculates the log returns for each final state of the pools over
    all the simulated paths.
    """
    x_T = np.zeros(params['batch_size']) # each element of x_T will be the final wealth of a path

    # In order to have the final wealth, we need to burn and swap the LP tokens
    # for each path. This is done by the     global log_returns method of the amm class.
    # The method takes all the LP tokens, burn them and swap coin-Y for coin-X.
    for k in range(params['batch_size']):
        x_T[k] = np.sum(final_pools_dists[k].burn_and_swap(l))

    # Calculate the initial wealth
    x_0 = np.sum(x0)

    # Calculate the log returns for each path
    log_returns = np.log(x_T) - np.log(x_0)

    return log_returns

def portfolio_evolution(initial_pools_dist, amm_instance, params):
    # Compute the actual tokens for each pool. The initial_pools_dist are the
    # weights of the pools. We need to multiply them by the initial wealth
    X0 = params['x_0'] * initial_pools_dist

    # Evaluate the number of LP tokens. This will be used to compute the returns
    l = amm_instance.swap_and_mint(X0)

    # Simulate the evolution of the pools (scenario simulation). We simulate params['batch_size'] paths, 
    # hence we will have params['batch_size'] amount of returns at the end.
    final_pools_dists, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = amm_instance.simulate(
        kappa=params['kappa'], p=params['p'], sigma=params['sigma'], T=params['T'], batch_size=params['batch_size'])

    # Calculate the log returns for each path
    global log_returns
    log_returns = calculate_log_returns(X0, final_pools_dists, l)

    return log_returns


def objective_function(parameters, amm_instance, params):
    """
    Objective function to minimize. It takes as input the initial wealth distribution
    ,the parameters of the model and the instance of the amm class. 
    It returns the CVaR of the final return distribution. Additionally, it evaluates
    the log returns and the probability of having a return greater than 0.05. It set these
    variables as global so that 'probability' can be used in the constraint_3 function and
    'log_returns' can be used to plot the distribution of returns for the best initial wealth
    distribution.
    """

    # Extract the parameters from the dictionary params
    kappa, p, sigma = params['kappa'], params['p'], params['sigma']
    T = params['T']
    batch_size = params['batch_size']
    x0 = params['x_0'] * initial_pools_dist

    # Evaluate the number of LP tokens
    l = amm_instance.swap_and_mint(x0)

    # Simulate the evolution of the pools.
    # final_pools_dist is a list of length batch_size. Each element of the list contains 
    # the final reserves of the pools for a given path. To access the final X reserve of
    # the i-th path you need to do final_pools_dist[i].Rx. Same for Ry.
    final_pools_dists, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = amm_instance.simulate(
        kappa=kappa, p=p, sigma=sigma, T=T, batch_size=batch_size)

    # Calculate the log returns for each path
    global log_returns
    log_returns = calculate_log_returns(x0, final_pools_dists, l)

    # Compute the probability of having a return greater than 0.05
    global probability
    probability = log_returns[log_returns > 0.05].shape[0] / log_returns.shape[0]

    # Calculate the CVaR of the final return distribution
    cvar = calculate_cvar(log_returns)

    return cvar

def objective_function_RU(parameters, amm_instance, params):
    '''Implement the objective function following Rockafellar and Uryasev (2000).'''

    # Extract the parameters to optimize from the parameters vector
    gamma = parameters[0]
    u = parameters[1:params['batch_size']+1]
    initial_pools_dist = parameters[-params['N_pools']:]

    # Evolve the portfolio
    global log_returns
    log_returns = portfolio_evolution(initial_pools_dist, amm_instance, params)

    # Compute the probability of having a return greater than 0.05
    global probability
    probability = log_returns[log_returns > 0.05].shape[0] / log_returns.shape[0]

    # Calculate the CVaR of the final return distribution
    cvar = calculate_cvar_RU(gamma, u, alpha=params['alpha'])

    return cvar


def constraint_1(x):
    return np.sum(x[-params['N_pools']:]) - 1

def constraint_2(x):
    global probability
    return probability - 0.7

# def constraint_3(x):
#     u = x[1:params['batch_size']+1]
#     return u

def constraint_4(x):
    global log_returns
    u = x[1:params['batch_size']+1]
    gamma = x[0]
    cond = u + log_returns + gamma
    return cond

def optimize_distribution(params):
    """
    Optimizes the distribution of wealth across liquidity pools to minimize CVaR,
    conditioned to P[final return > 0.05]>0.7.

    Args:
    - amm_instance (amm): Instance of the amm class.
    - params (dict): Parameters for the amm and optimization.

    Returns:
    - dict: Optimal weights and corresponding CVaR.
    """

    # Global variables to store the log returns and the
    # probability of having a return greater than 0.05
    global probability
    global log_returns
    log_returns, probability = 0, 0

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': constraint_1},
                {'type': 'ineq', 'fun': constraint_2},
                {'type': 'ineq', 'fun': constraint_4}]
    bounds_initial_dist = [(0, 1) for i in range(params['N_pools'])]
    bounds_u = [(0, None) for i in range(params['batch_size'])]
    bounds = [(0, 1), *bounds_u, *bounds_initial_dist]
    
    # Instantiate the amm class
    amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])
    
    # Callback function to print the current CVaR and the current parameters
    def callback_function(x, *args):
        current_cvar = objective_function_RU(x, amm_instance, params)
        logging.info(f"Current initial_dist: {x[-params['N_pools']:]}")
        logging.info(f"Current probability: {probability}")
        logging.info(f"Current VaR:{x[0]}")
        logging.info(f"Current CVaR: {current_cvar}\n")

    # The following while loop is used to check if the initial distribution of wealth
    # across pools is feasible. If it is not, a new one is generated.
    cond = True
    while cond == True:
        # Initial distribution of wealth across pools
        random_vector = np.random.uniform(0, 100, params['N_pools'])
        initial_pools_dist = random_vector / np.sum(random_vector)
        try:
            amm_instance.swap_and_mint(initial_pools_dist*params['x_0'], quote=True)
            cond = False
        except ValueError as e:
            logging.info(f"Error: {e}")
        
    gamma = np.random.uniform(0, 1)
    u = np.random.uniform(0, 1, params['batch_size'])
    initial_guess = np.array([gamma, *u, *initial_pools_dist])
    logging.info(f"Initial guess:\n\t{initial_guess}")

    # Optimization procedure
    logging.info(f"Optimization method: Rockafellar and Uryasev (2000). Start...")
    result = minimize(objective_function_RU, initial_guess, args=(amm_instance, params),
                method='trust-constr', constraints=constraints, bounds=bounds, callback=callback_function)

    logging.info(f"Results:\n\t{result}")
    print(result)

    return result.x

def simulation_plots(res, params):
    """
    Plots the evolution of the reserves, the price and the returns for a given
    initial distribution of wealth across pools.
    """
    x_0 = res[-params['N_pools']:]
    var = res[0]
    # Initialize the pools
    amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])
    # Evaluate the number of LP tokens
    l = amm_instance.swap_and_mint(x_0)

    # Simulate the evolution of the pools.
    # final_pools_dist is a list of length batch_size. Each element of the list contains 
    # the final reserves of the pools for a given path. To access the final X reserve of
    # the i-th path you need to do final_pools_dist[i].Rx. Same for Ry.
    x_T, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = amm_instance.simulate(kappa=params['kappa'], p=params['p'], sigma=params['sigma'], T=params['T'], batch_size=1000)
    log_returns = calculate_log_returns(x_0, x_T, l)
    cvar = calculate_cvar(log_returns)

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
    plt.savefig('pools.png')

    # Plot the distribution of the returns
    plt.figure(figsize=(10, 8), tight_layout=True)
    plt.hist(log_returns, bins=50, alpha=0.7)
    plt.axvline(cvar, color='r', linestyle='dashed', linewidth=1, label='CVaR')
    plt.axvline(var, color='b', linestyle='dashed', linewidth=1, label='VaR')
    plt.axvline(0.05, color='g', linestyle='dashed', linewidth=1, label=r'$\xi$')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.savefig('returns.png')

    plt.show()
