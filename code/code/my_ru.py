
import numpy as np
from amm import amm
from params import params
from scipy.optimize import minimize
from utils import calculate_log_returns, constraint_1, portfolio_evolution

# Ignore future warnings
import warnings
warnings.simplefilter(action='ignore')

# Some variables for the optimization
options = {'maxiter': 1000, 'ftol': 1e-8}
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x)-1}]
bounds_initial_dist = [(1e-5, 1) for i in range(params['N_pools'])]

# Then, minimize the actual loss function
np.random.seed(params['seed'])
# Global variables to store the log returns and the
# probability of having a return greater than 0.05
log_returns, probability = 0, 0

# Constraints, bounds and options for the optimization
constraints = [{'type': 'eq', 'fun': constraint_1}]
bounds_initial_dist = [(1e-5, 1) for i in range(params['N_pools'])]
options = {'maxiter': 1000, 'ftol': 1e-8}

# Instantiate the amm class
amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])

# The following while loop is used to check if the initial distribution of wealth
# across pools is feasible. If it is not, a new one is generated.
cond = True
while cond == True:
    # Initial distribution of wealth across pools
    random_vector = np.random.uniform(0, 100, params['N_pools'])
    initial_guess = random_vector / np.sum(random_vector)
    try:
        l = amm_instance.swap_and_mint(initial_guess*params['x_0'])
        cond = False
        end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t =\
            amm_instance.simulate(
                kappa=params['kappa'],
                p=params['p'],
                sigma=params['sigma'],
                T=params['T'],
                batch_size=params['batch_size'])
        log_ret = calculate_log_returns(initial_guess*params['x_0'], end_pools, l)
        constraint= len(log_ret[ log_ret>params['zeta'] ]) / len(log_ret) - params['q']
        if constraint < 0:
            cond = True
    except ValueError as e:
        print(f"Error: {e}")

# Optimization procedure
result = minimize(portfolio_evolution, initial_guess, args=(amm_instance, params), tol=1e-6,
        method='SLSQP', bounds=bounds_initial_dist, constraints=constraints, options=options)
print(result)
