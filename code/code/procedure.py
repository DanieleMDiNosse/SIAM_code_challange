
import time
import datetime
import warnings
import numpy as np
from amm import amm
from params import params
from scipy.optimize import minimize
import multiprocessing
from utils import portfolio_evolution, simulation_plots, worker, KernelRidge_Warper

warnings.simplefilter(action='ignore')

print(f'Start: {datetime.datetime.now()}')

start_time0 = time.time()
print('Generating random numbers for the simulation...')
np.random.seed(params['seed'])
amm = amm(params['Rx0'], params['Ry0'], params['phi'])
random_numbers = amm.save_random_numbers(
    kappa= params['kappa'],
    T = params['T'], 
    batch_size = params['batch_size'])
print(f'Done in {time.time()-start_time0:.2f} seconds')

#Initialize the lists of data
x_data, y_data = list(), list()
# Iterates over random seeds

# Select one of the following hp combinations from kernel_regression.py
# hp = {'kernel': 'additive_chi2', 'alpha': 1, 'n_points':32}
# hp = {'kernel': 'additive_chi2', 'alpha': 0.1, 'n_points':10}
# hp = {'kernel': 'additive_chi2', 'alpha': 0.01, 'n_points':10}
# hp = {'kernel': 'additive_chi2', 'alpha': 0.01, 'n_points':32}
hp = {'kernel': 'additive_chi2', 'alpha': 0.01}

start_time = time.time()
print('Estimating initial point using KRR...')
# Create a queue to collect results
result_queue = multiprocessing.Queue()

# Define the range for each process
n_points_per_process = 5
ranges = [(0, n_points_per_process), (n_points_per_process, 10)]

# Create and start processes
processes = []
for start, end in ranges:
    p = multiprocessing.Process(target=worker, args=(start, end, params, random_numbers, result_queue))
    processes.append(p)
    p.start()

# Wait for processes to complete
for p in processes:
    p.join()

# Collect results
while not result_queue.empty():
    local_x, local_y = result_queue.get()
    x_data.extend(local_x)
    y_data.extend(local_y)

# Some variables for the optimization
options = {'maxiter': 1000, 'ftol': 1e-8}
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x)-1}]
bounds_initial_dist = [(1e-5, 1) for i in range(params['N_pools'])]

# Convert to numpy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)

# Fit the model
krr = KernelRidge_Warper(hp)
krr.fit(x_data, y_data)

# Find the minimum and save it
result = minimize(lambda x: krr.predict(x), np.array([1/6]*6),
                method='SLSQP', bounds=bounds_initial_dist,
                constraints=constraints, options=options, tol=1e-8)
print(f'Done in {time.time() - start_time:.2f} seconds')
print('The starting point for the second step is:', result.x)
print('Loss function approximated:', result.fun)
print('Loss function real:', portfolio_evolution(result.x, random_numbers, params))

# Then, minimize the actual loss function
np.random.seed(params['seed'])

options = {'maxiter': 1000, 'ftol': 1e-6}
# Optimization procedure
start_time = time.time()
print('Optimizing cVaR...')
result = minimize(portfolio_evolution, result.x, args=(random_numbers, params),
                  method='SLSQP', bounds=bounds_initial_dist,
                  constraints=constraints, tol=1e-6, options=options)
print(f'Done in {time.time() - start_time:.2f} seconds')
print(result)
print(portfolio_evolution(result.x, random_numbers, params))
print(f'Total time needed {time.time() - start_time0:.2f} seconds')
print(f'End: {datetime.datetime.now()}')

simulation_plots(result.x, random_numbers, params)
