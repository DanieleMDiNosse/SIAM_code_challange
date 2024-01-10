
#%% Importing

import copy
import pickle
import random
import numpy as np
from amm import amm
from params import params
from tqdm.auto import tqdm
import multiprocessing as mp
from utils import calculate_cvar, calculate_log_returns

'''
Summary of the best results
MINIMUM rgs
cvar = 

Top 5%
cvar = 

Top 1%
cvar = 

Equi-weighted
cvar = 0.005193897750906791


Daniele
cvar = 0.0036268


Mio IV
cvar = 0.004570863305932949

'''

OUTPUT_FOLDER = '/home/garo/Desktop/Lavoro_Studio/[SIAG] Challenge/SIAM_code_challange/code/temp_results'

# %% Define the function for the optimization

import numpy as np
from utils import calculate_cvar, calculate_log_returns

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

    constraint= len(log_ret[ log_ret>params['zeta'] ]) / len(log_ret) - params['q']
    if constraint >= 0:
        # If the constraint is satisfied, return VaR and CVaR
        var = np.quantile(-log_ret, params['alpha'])
        cvar = np.mean(-log_ret[-log_ret >= var])
        cvar = np.mean(-log_ret[-log_ret >= var])
    else: #Otherwise, return np.inf or np.nan
        if ret_inf:
            cvar = np.nan
        else:
            cvar = np.inf
    return constraint, var, cvar

def worker_process(params, start_seed, end_seed, folder):
    worker_save_points, worker_save_cvar, worker_save_var, worker_save_con =\
        list(), list(), list(), list()

    for rgs_seed in range(start_seed, end_seed):
        np.random.seed(rgs_seed)

        #Randomly draw the vector of weights theta and refularize it to have sum=1
        theta = list()
        for _ in range(len(params['Rx0'])):
            theta.append( np.random.uniform() )
        theta = np.array(theta) / np.sum(theta)

        worker_save_points.append(theta)
        con, var, cvar = target_4_opt(theta, params, ret_inf=False)
        worker_save_cvar.append( cvar )
        worker_save_var.append( var )
        worker_save_con.append( con )
    
    '''# Put results in output queue
    output_queue.put((worker_save_points, worker_save_cvar,
                      worker_save_var, worker_save_con))'''
    
    with open(f'{folder}/rgs_output{os.getpid()}.pickle', 'wb') as f:
        pickle.dump({'points':worker_save_points,
                    'cvar':worker_save_cvar,
                    'var':worker_save_var,
                    'constraint':worker_save_con}, f)
    print('Fine', n_it)

#%% Random Grid Search

num_workers = 6
seeds_per_launch = 12000
seeds_per_worker = seeds_per_launch // num_workers

'''output_queue = mp.Queue()'''

for n_it in tqdm(range(1), 'Computing grid search...'):
    # Create and start worker processes
    workers = list()
    for i in range(num_workers):
        start_seed = seeds_per_launch*n_it + i * seeds_per_worker
        end_seed = start_seed + seeds_per_worker
        worker = mp.Process(target=worker_process,
                            args=(params, start_seed, end_seed, OUTPUT_FOLDER))
        workers.append(worker)
        worker.start()

    # Gather results from workers
    for worker in workers:
        worker.join()

save_points = list() #Initialize lists of result
save_cvar = list() 
save_var = list() 
save_con = list()

main_word = 'rgs_output'
for file in os.listdir(OUTPUT_FOLDER):
    if len(file) > len(main_word+'.pickle') and file[:len(main_word)] == main_word:
        with open(f'{OUTPUT_FOLDER}/{file}', 'rb') as f:
            worker_save_points, worker_save_cvar, worker_save_var, worker_save_con =\
                pickle.load(f)
        save_points.extend(worker_save_points)
        save_cvar.extend(worker_save_cvar)
        save_var.extend(worker_save_var)
        save_con.extend(worker_save_con)
        os.remove(f'{OUTPUT_FOLDER}/{file}')

# Save results
with open(f'{OUTPUT_FOLDER}/rgs_output.pickle', 'wb') as f:
    pickle.dump({'points':save_points,
                'cvar':save_cvar,
                'var':save_var,
                'constraint':save_con}, f)
