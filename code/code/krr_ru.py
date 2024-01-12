
'''
Select one of the following hp combinations from kernel_regression.py
hp = {'kernel': 'additive_chi2', 'alpha': 1, 'n_points':32}
hp = {'kernel': 'additive_chi2', 'alpha': 0.1, 'n_points':10}
hp = {'kernel': 'additive_chi2', 'alpha': 0.01, 'n_points':10}
hp = {'kernel': 'additive_chi2', 'alpha': 0.01, 'n_points':32}
'''

import time
import pickle
import numpy as np
from amm import amm
from params import params
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
from utils import calculate_cvar, calculate_log_returns, constraint_1, portfolio_evolution

# Ignore future warnings
import warnings
warnings.simplefilter(action='ignore')

DATA_FOLDER = '../'

def target_4_opt(theta, params, ret_inf=False, full_output=True):
    '''
    Target function for the optimization.
    INPUT:
        - theta: np.array containing the weights of the portfolio
        - params: dict containing the parameters of the simulation
        - ret_inf: bool, if True when the constraint E[ r>zeta ] > 0.7 is not
            satisfied the output is np.inf; otherwise is np.nan
        - full_output: bool, whenever to include also the constraint in the output
    OUTPUT:
        - cvar = If the constraint E[ r>zeta ] > 0.7 is satisfied, return CVaR.
            Otherwise, the result is either np.nan or np.inf, according to ret_inf
        - constraint = E[ r>zeta ] - 0.7
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
            kappa=params['kappa'],
            p=params['p'],
            sigma=params['sigma'],
            T=params['T'],
            batch_size=params['batch_size'])
    # Compute the log returns
    log_ret = calculate_log_returns(xs_0, end_pools, l)

    constraint= len(log_ret[ log_ret>params['zeta'] ]) / len(log_ret) - 0.7
    cvar, var = calculate_cvar(log_ret) #If the constraint is satisfied, return CVaR
    return constraint, cvar, var

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

x_data, y_data = list(), list() #Initialize the lists of data
# Iterates over random seeds
for rgs_seed in range(hp.pop('n_points')):
    np.random.seed(rgs_seed)

    #Randomly draw the vector of weights theta and refularize it to have sum=1
    theta = list()
    for _ in range(len(params['Rx0'])):
        theta.append( np.random.uniform() )
    theta = np.array(theta) / np.sum(theta)

    # Compute the cvar
    _, cvar, _ = target_4_opt(theta, params, ret_inf=False)

    # Store the data
    x_data.append(theta)
    y_data.append(cvar)

# Some variables for the optimization
options = {'maxiter': 1000, 'ftol': 1e-8}
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x)-1}]
bounds_initial_dist = [(1e-5, 1) for i in range(params['N_pools'])]

# Select the points
x_data = np.array(x_data)
y_data = np.array(y_data)
# Fit the model
krr = KernelRidge_Warper(hp)
krr.fit(x_data, y_data)
# Find the minimum and save it
result = minimize(lambda x: krr.predict(x), np.array([1/6]*6),
                method='SLSQP', bounds=bounds_initial_dist,
                constraints=constraints, options=options, tol=1e-8)
print('Finished the first part of the optimization')
print('The starting point for the second step is:', result.x)
print('Loss function approximated:', result.fun)
print('Loss function real:', target_4_opt(result.x, params)[1])

# Then, minimize the actual loss function
np.random.seed(params['seed'])
# Global variables to store the log returns and the
# probability of having a return greater than 0.05
log_returns, probability = 0, 0

# Instantiate the amm class
amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])

options = {'maxiter': 1000, 'ftol': 1e-6}
# Optimization procedure
result = minimize(portfolio_evolution, result.x, args=(amm_instance, params),
                  method='SLSQP', bounds=bounds_initial_dist,
                  constraints=constraints, tol=1e-6, options=options)
print(result)
target_4_opt(result.x, params)
