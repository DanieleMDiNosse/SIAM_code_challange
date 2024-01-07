
#%%

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from amm import amm
from params import params
import argparse

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

#%% I - As Daniele

logging.info(f"Optimization method: Rockafellar and Uryasev (2000). Start...")

result = minimize(objective_function_RU, initial_guess, args=(amm_instance, params),
            method='trust-constr', constraints=constraints, bounds=bounds)# callback=callback_function)
theta = np.array(res.x[1:]) / np.sum(res.x[1:])
print(theta)
target_4_opt(theta, params)

#%% II - SLSQP minimizer (incremental tolerance)

from scipy.optimize import minimize, LinearConstraint

constr = LinearConstraint(
    np.concatenate([np.eye(len(initial_guess))], axis=0),
    np.concatenate([np.ones(len(initial_guess))*1e-5], axis=0),
    np.concatenate([[0.1], np.ones(len(initial_guess)-1)], axis=0),
    keep_feasible=True)

# Step 1
res = minimize(objective_function_RU, initial_guess, args=(amm_instance, params),
               method='SLSQP', constraints=constr, options={'disp':False}
               ) #Minimize to find the next theta step
theta = np.array(res.x[1:]) / np.sum(res.x[1:])
print(theta)
target_4_opt(theta, params)
# (0.139, (0.00449591600270697, -0.03243495952278879))

# Step 2
res = minimize(objective_function_RU, res.x, args=(amm_instance, params),
               method='SLSQP', constraints=constr, options={'disp':False}, tol=1e-6
               ) #Minimize to find the next theta step
theta = np.array(res.x[1:]) / np.sum(res.x[1:])
print(theta)
target_4_opt(theta, params)
# (0.139, (0.004495916606229051, -0.03243495499939255))

# Step 3
res = minimize(objective_function_RU, res.x, args=(amm_instance, params),
               method='SLSQP', constraints=constr, options={'disp':False}, tol=1e-10
               ) #Minimize to find the next theta step
theta = np.array(res.x[1:]) / np.sum(res.x[1:])
print(theta)
target_4_opt(theta, params)
'''
Questo non va
'''

#%% III - trust-constr minimization with increasing T

for n_per in range(10, 70, 10):
    temp_params = params.copy()
    temp_params['T'] = n_per
    result = minimize(objective_function_RU, initial_guess, args=(amm_instance, params),
                method='trust-constr', constraints=constraints, bounds=bounds)# callback=callback_function)
    initial_guess = res.x
    theta = np.array(res.x[1:]) / np.sum(res.x[1:])
    print('I am considering', n_per, 'periods in the simulation.')
    print(theta)
    target_4_opt(theta, params)
    print('\n')
