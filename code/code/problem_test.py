
#%%

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from amm import amm
from params import params
import argparse

def objective_function_RU(parameters, amm_instance, params):
    '''Implement the objective function following Rockafellar and Uryasev (2000).'''

    # Extract the parameters to optimize from the parameters vector
    VaR = parameters[0]
    initial_pools_dist = parameters[-params['N_pools']:]

    # Evolve the portfolio
    global log_returns

    log_returns = portfolio_evolution(initial_pools_dist, amm_instance, params)

    # Compute the probability of having a return greater than 0.05
    global probability
    probability = log_returns[log_returns > 0.05].shape[0] / log_returns.shape[0]

    # Calculate the CVaR of the final return distribution
    cvar = calculate_cvar_RU(VaR, -log_returns, alpha=params['alpha'])

    logging.info(f"Random number:{np.random.normal()}")
    logging.info(f"Current initial_dist: {initial_pools_dist}")
    logging.info(f"Current CVaR: {cvar}")
    logging.info(f"Current probability: {probability}")
    logging.info(f"Current VaR:{VaR}")
    logging.info(f"Current returns mean:{np.mean(log_returns)}\n")

    return cvar

parser = argparse.ArgumentParser(description='Optimize the initial wealth distribution')
parser.add_argument('-s', '--simulate', type=int, default=0, help='Simulate the model with the optimal initial wealth distribution (1) or not (0)')
#args = parser.parse_args()

_ = logging_config('opt')
np.random.seed(params['seed'])

log_returns, probability = 0, 0

# Constraints and bounds
constraints = [{'type': 'eq', 'fun': constraint_1},
            {'type': 'ineq', 'fun': constraint_2}]
            #{'type': 'ineq', 'fun': constraint_3}]
bounds_initial_dist = [(0, 1) for i in range(params['N_pools'])]
bounds = [(0, 0.1), *bounds_initial_dist]

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
    
initial_VaR = np.random.uniform(0, 0.1)
initial_guess = np.array([initial_VaR, *initial_pools_dist])
logging.info(f"Initial guess:\n\t{initial_guess}\n")

print('This is amm_instance.Rx before launching objective_function_RU')
print(amm_instance.Rx)
objective_function_RU(initial_guess, amm_instance, params)
print('And this is after the iteration')
print(amm_instance.Rx)
print('That is, amm_instance changes at each iteration while it should be fixed!!!')
print('\n\n')

import copy
print('Now, we try to give as input copy.deepcopy(amm_instance). Before')
print(amm_instance.Rx)
objective_function_RU(initial_guess, copy.deepcopy(amm_instance), params)
print('And after')
print(amm_instance.Rx)
print('It is the same!!!')