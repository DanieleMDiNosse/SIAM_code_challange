import pickle
import random
import numpy as np
from amm import amm
from params import params
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
from utils import calculate_cvar, calculate_log_returns, logging_config
import multiprocessing as mp

OUTPUT_FOLDER = 'output'

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
    if constraint >= 0:
        cvar = calculate_cvar(log_ret) #If the constraint is satisfied, return CVaR
    else: #Otherwise, return np.inf or np.nan
        if ret_inf:
            cvar = np.nan
        else:
            cvar = np.inf
    return constraint, cvar

def worker_process(start_seed, end_seed, output_queue):
    np.random.seed()  # Ensure each process has a different seed
    worker_save_points, worker_save_cvar, worker_save_con = [], [], []

    for rgs_seed in range(start_seed, end_seed):
        logging.info(f'Random seed: {os.getpid()}:{rgs_seed}')
        np.random.seed(rgs_seed)

        #Randomly draw the vector of weights theta and refularize it to have sum=1
        theta = list()
        for _ in range(len(params['Rx0'])):
            theta.append( np.random.uniform() )
        theta = np.array(theta) / np.sum(theta)

        worker_save_points.append(theta)
        con, cvar = target_4_opt(theta, params, ret_inf=False)
        worker_save_cvar.append( cvar )
        worker_save_con.append( con )
    
    # Put results in output queue
    output_queue.put((worker_save_points, worker_save_cvar, worker_save_con))

if __name__ == '__main__':
    job_id = logging_config('rgs')

    #Initialize lists of result
    save_points = list()
    save_cvar = list() 
    save_con = list()

    # Number of workers and range of seeds for each worker
    num_workers = 15
    seeds_per_worker = 10000 // num_workers

     # Create a queue to collect results from workers
    output_queue = mp.Queue()

    # Create and start worker processes
    workers = []
    for i in range(num_workers):
        start_seed = i * seeds_per_worker
        end_seed = start_seed + seeds_per_worker
        worker = mp.Process(target=worker_process, args=(start_seed, end_seed, output_queue))
        workers.append(worker)
        logging.info(f'Starting worker {i}...')
        worker.start()

    # Gather results from workers
    for worker in workers:
        logging.info('Gathering results from worker...')
        worker.join()
        worker_save_points, worker_save_cvar, worker_save_con = output_queue.get()
        save_points.extend(worker_save_points)
        save_cvar.extend(worker_save_cvar)
        save_con.extend(worker_save_con)

    # Save results
    logging.info(f'Saving results...')
    with open(f'{OUTPUT_FOLDER}/rgs_output_{job_id}.pickle', 'wb') as f:
        pickle.dump({'points':save_points,
                    'cvar':save_cvar,
                    'constraint':save_con}, f)
    # print the minimun value of the CVaR and the corresponding weights
    logging.info(f'Minimum CVaR: {np.min(save_cvar)}')
    logging.info(f'Weights: {save_points[ np.argmin(save_cvar)]}')