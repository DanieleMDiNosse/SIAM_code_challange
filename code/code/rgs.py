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
from utils import calculate_cvar, calculate_log_returns
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
        logging.info(f'Random seed: {rgs_seed}')
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
    if os.getenv("PBS_JOBID") != None:
        job_id = os.getenv("PBS_JOBID")
    else:
        job_id = os.getpid()

    logging.basicConfig(filename=f'output/output_{job_id}.log', format='%(message)s', level=logging.INFO)
    save_points = list() #Initialize lists of result
    save_cvar = list() 
    save_con = list()

    # Number of workers and range of seeds for each worker
    num_workers = 15
    seeds_per_worker = 10000 // num_workers

     # Create a queue to collect results from workers
    output_queue = mp.Queue()

    # Iterate over the random seed
    for rgs_seed in range(10000):
        logging.info(f'Random seed: {rgs_seed}')
        np.random.seed(rgs_seed)

        #Randomly draw the vector of weights theta and refularize it to have sum=1
        theta = list()
        for _ in range(len(params['Rx0'])):
            theta.append( np.random.uniform() )
        theta = np.array(theta) / np.sum(theta)

        save_points.append(theta)
        con, cvar = target_4_opt(theta, params, ret_inf=False)
        save_cvar.append( cvar )
        save_con.append( con )

    # Create and start worker processes
    workers = []
    for i in range(num_workers):
        start_seed = i * seeds_per_worker
        end_seed = start_seed + seeds_per_worker
        worker = mp.Process(target=worker_process, args=(start_seed, end_seed, output_queue))
        workers.append(worker)
        worker.start()

    # Gather results from workers
    for worker in workers:
        worker.join()
        worker_save_points, worker_save_cvar, worker_save_con = output_queue.get()
        save_points.extend(worker_save_points)
        save_cvar.extend(worker_save_cvar)
        save_con.extend(worker_save_con)

    with open(f'{OUTPUT_FOLDER}/rgs_output.pickle', 'wb') as f:
        pickle.dump({'points':save_points,
                    'cvar':save_cvar,
                    'constraint':save_con}, f)
    # print the minimun value of the CVaR and the corresponding weights
    logging.info(f'Minimum CVaR: {np.min(save_cvar)}')
    logging.info(f'Weights: {save_points[ np.argmin(save_cvar)]}')
    exit()
    sns.set_theme()

    #Load the data
    with open(f'{OUTPUT_FOLDER}/rgs_output.pickle', 'rb') as f:
        opt_res = pickle.load(f)
        
    # Plot the cvar as a function of the constraint. I'm trying to understand
    #   whenever the constraint is active (i.e. cvar minimum is on the boundary con=0)

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(x=np.array(opt_res['save_points'])[ ~ np.isnan(opt_res['cvar']) ],
                    y=np.array(opt_res['cvar'])[ ~ np.isnan(opt_res['cvar']) ], ax = ax)
    ax.axhline(0.03121627321033861, color=sns.color_palette()[1], linestyle='--', label='Equi-weighted portfolio')
    ax.set_xlabel(r'Initial dist')
    ax.set_ylabel(r'$CVaR_\alpha$')
    ax.legend()
    plt.show()

    # Histplot of the CVaR. Maybe, in the future it can be useful to contextualize
    #   the performance of out approach

    fig, ax = plt.subplots(1, 1)
    sns.histplot(ax = ax, x=opt_res['cvar'], kde=True)
    ax.axvline(0.03121627321033861, color=sns.color_palette()[1], linestyle='--', label='Equi-weighted portfolio')
    ax.set_xlabel(r'$CVaR_\alpha$')
    ax.legend()
    plt.show()

    # My first idea is to use an iterative algorithm, where we locally assume the theta
    #   impact on the pool dynamic is negligible, and it only after portfolio weights.

    approx = True #Determine whenever to compute the true or the approximated value

    theta = [1]*6
    theta = np.array(theta) / np.sum(theta)

    np.random.seed(params['seed']) #Fix the seed for the next operations

    #Initialize the pools
    Rx0 = params['Rx0']
    Ry0 = params['Ry0']
    phi = params['phi']
    pools = amm(Rx=Rx0, Ry=Ry0, phi=phi)

    #The amount to invest on each pool is given by the weight in theta mult by the total capital
    xs_0 = params['x_0']*theta
    # Obtain the quote for the swap and mint
    l = pools.swap_and_mint(xs_0) #Swap and mint to obtain the quote

    if approx:
        pools = amm(Rx=Rx0, Ry=Ry0, phi=phi) #Reinitialize the pools to neglect the changes

    # Simulate 1000 paths of trading in the pools
    end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t =\
        pools.simulate(
            kappa=params['kappa'],
            p=params['p'],
            sigma=params['sigma'],
            T=params['T'],
            batch_size=params['batch_size'])

    if approx:
        for k in range(len(end_pools)):
            end_pools[k].l = l.copy()

    #Compute the return
    log_ret = calculate_log_returns(xs_0, end_pools, l)
    constraint= len(log_ret[ log_ret>params['zeta'] ]) / len(log_ret) - 0.7
    cvar = calculate_cvar(log_ret)
    if approx:
        print('Approximated value: Constraint = ', round(constraint, 5),
            '     CVaR', round(cvar, 5))
    else:
        print('Actual value: Constraint = ', round(constraint, 5),
            '     CVaR', round(cvar, 5))
