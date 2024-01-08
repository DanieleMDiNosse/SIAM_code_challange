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

# if os.path.exists('output.log'):
#     os.remove('output.log')
logging.basicConfig(filename=f'output_{job_id}.log', format='%(message)s', level=logging.INFO)

def calculate_cvar(log_returns):
    """
    Calculate the CVaR of a set of returns.
    """
    global params

    quantile = np.quantile(-log_returns, params['alpha'])
    cvar = np.mean(-log_returns[-log_returns >= quantile])
    return cvar

def calculate_cvar_RU(log_returns, alpha, beta):
    """
    Calculate the CVaR of a set of returns following Rockafellar and Uryasev.
    """

    global u
    u = -log_returns - alpha
    cvar_RU = alpha + 1 / ((1 - beta) * log_returns.shape[0]) * np.sum(u)
    return cvar_RU, u

def calculate_log_returns(x0, final_pools_dists, l):
    """
    In the simulate method of the amm class, a  certain number of trading paths are
    generated. Consequently, for each path we have a final log return.
    This function calculates the log returns for each final state of the pools over
    all the simulated paths.
    """

    batch_size = len(final_pools_dists) # number of simulated paths
    x_T = np.zeros(batch_size) # each element of x_T will be the final wealth of a path

    # In order to have the final wealth, we need to burn and swap the LP tokens
    # for each path. This is done by the burn_and_swap method of the amm class.
    # The method takes all the LP tokens, burn them and swap coin-Y for coin-X.
    for k in range(batch_size):
        x_T[k] = np.sum(final_pools_dists[k].burn_and_swap(l))

    # Calculate the initial wealth
    x_0 = np.sum(x0)

    # Calculate the log returns for each path
    log_returns = np.log(x_T) - np.log(x_0)

    return log_returns

def objective_function(initial_pools_dist, amm_instance, params):
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
    alpha = parameters[0]
    initial_pools_dist = parameters[1:]

    beta = params['alpha']
    q = params['batch_size']

    x0 = params['x_0'] * initial_pools_dist

    l = amm_instance.swap_and_mint(x0)

    final_pools_dists, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = amm_instance.simulate(
        kappa=params['kappa'], p=params['p'], sigma=params['sigma'], T=params['T'], batch_size=q)

    # Calculate the log returns for each path
    global log_returns
    log_returns = calculate_log_returns(x0, final_pools_dists, l)

    # Compute the probability of having a return greater than 0.05
    global probability
    probability = log_returns[log_returns > 0.05].shape[0] / log_returns.shape[0]

    # Calculate the CVaR of the final return distribution
    global u
    cvar, u = calculate_cvar_RU(log_returns, alpha, beta)

    return cvar


def constraint_1(x):
    return np.sum(x) - 1

def constraint_2(x):
    global probability
    return probability - 0.7

def constraint_3(x):
    global u
    return u

def constraint_4(x):
    global u
    global log_returns
    c = x[0] + u + log_returns
    return c

def optimize_distribution(params, method='RockafellarUryasev'):
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
    log_returns = 0
    probability = 0

    # Constraints and bounds
    if method == 'RockafellarUryasev':
        constraints = [{'type': 'eq', 'fun': constraint_1},
                   {'type': 'ineq', 'fun': constraint_2},
                   {'type': 'ineq', 'fun': constraint_3},
                   {'type': 'ineq', 'fun': constraint_4}]
        bounds = [(0, 1) for i in range(params['N_pools'])]
        bounds.insert(0, (None, None))
    else:
        constraints = [{'type': 'eq', 'fun': constraint_1},
                   {'type': 'ineq', 'fun': constraint_2}]
        bounds = [(0, 1) for i in range(params['N_pools'])]
    
    # Instantiate the amm class
    amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])
    
    # Callback function to print the current CVaR and the current parameters
    def callback_function(x, state):
        # `state` is a placeholder for the second argument passed by `scipy.optimize.minimize`. 
        # It is used by trust-constr.
        if method == 'RockafellarUryasev':
            current_cvar = objective_function_RU(x, amm_instance, params)
        else:
            current_cvar = objective_function(x, amm_instance, params)
        logging.info(f"Current parameters: {x}")
        logging.info(f"Current CVaR: {current_cvar}\n")
    
    # Options for SLSQP
    options = {
        'maxiter': 1000,  # Maximum number of iterations (default: 100)
        'ftol': 1e-9,    # Precision of the objective function value (default: 1e-6)
        'eps': 1.5e-8,    # Step size used for numerical approximation of the Jacobian (default is sqrt(eps_machine))
        }

    cond = True
    while cond == True:
        # Initial distribution of wealth across pools
        random_vector = np.random.uniform(0, 100, params['N_pools'])
        initial_pools_dist = random_vector / np.sum(random_vector)
        try:
            # Check it the guess is feasible
            amm_instance.swap_and_mint(random_vector*params['x_0'], quote=True)
            logging.info(f"Initial pools distribution:\n\t{initial_pools_dist}")
            # Optimization procedure
            if method == 'RockafellarUryasev':
                result = minimize(objective_function_RU, np.insert(initial_pools_dist, 0, 0), args=(amm_instance, params),
                            method='SLSQP', constraints=constraints, bounds=bounds, callback=callback_function)#, options=options)
            else:
                result = minimize(objective_function, initial_pools_dist, args=(amm_instance, params),
                            method='trust-constr', constraints=constraints, bounds=bounds, callback=callback_function)#, options=options)
            cond = False
        except ValueError as e:
            logging.info(f"Error: {e}")

    logging.info(f"Results:\n\t{result}")
    print(result)

    return result.x

def target_4_opt(theta, params, ret_inf=False, full_output=True):
    '''
    Target function for the optimization.
    INPUT:
        - theta: np.array containing the weights of the portfolio
        - params: dict containing the parameters of the simulation
        - ret_inf: bool, if True when the constraint P[ r>zeta ] > 0.7 is not
            satisfied the output is np.inf; otherwise is np.nan
        - full_output: bool, whenever to include also the constraint in the output
    OUTPUT:
        - cvar = If the constraint P[ r>zeta ] > 0.7 is satisfied, return CVaR.
            Otherwise, the result is either np.nan or np.inf, according to ret_inf
        - constraint = P[ r>zeta ] - 0.7
    '''
    np.random.seed(params['seed']) #Fix the seed for the next operations

    #Initialize the pools
    Rx0 = params['Rx0']
    Ry0 = params['Ry0']
    phi = params['phi']
    pools = amm(Rx=Rx0, Ry=Ry0, phi=phi)

    #The amount to invest on each pool is given by the weight in theta mult by the total capital
    xs_0 = params['x_0']*theta
    # Swap and mint
    l = pools.swap_and_mint(xs_0)

    # Simulate 1000 paths of trading in the pools
    end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t =\
        pools.simulate(
            kappa=np.array(params['kappa']),
            p=np.array(params['p']),
            sigma=np.array(params['sigma']),
            T=params['T'],
            batch_size=params['batch_size'])
    # Compute the log returns
    log_ret = calculate_log_returns(xs_0, end_pools, l)

    constraint= len(log_ret[ log_ret>params['zeta'] ]) / len(log_ret) - 0.7
    if constraint >= 0:
        cvar = calculate_cvar(log_ret) #If the constraint is satisfied, return CVaR
    else: #Otherwise, return np.inf or np.nan
        if ret_inf:
            cvar = np.nan
        else:
            cvar = np.inf
    return constraint, cvar

def simulation_plots(x_0, params):
    """
    Plots the evolution of the reserves, the price and the returns for a given
    initial distribution of wealth across pools.
    """

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
    plt.hist(log_returns, bins=50)
    plt.axvline(cvar, color='r', linestyle='dashed', linewidth=1, label='CVaR')
    plt.axvline(0.05, color='g', linestyle='dashed', linewidth=1, label=r'$\xi$')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.savefig('returns.png')

    plt.show()
