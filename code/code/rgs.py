import pickle
import random
import numpy as np
from amm import amm
from params import params
from tqdm.auto import tqdm
from utils import calculate_cvar, calculate_log_returns, logging_config
import logging
import os

OUTPUT_FOLDER = 'output'

import numpy as np
from utils import calculate_cvar, calculate_log_returns

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
        cvar, var = calculate_cvar(log_ret) #If the constraint is satisfied, return CVaR
    else: #Otherwise, return np.inf or np.nan
        if ret_inf:
            cvar = np.nan
        else:
            cvar = np.inf
    return constraint, cvar

if __name__ == '__main__':
    save_points = list() #Initialize lists of result
    save_cvar = list() 
    save_con = list()

    c = 0
    logging_config('rgs') #Initialize the logger
    # Iterate over the random seed
    for rgs_seed in range(20000):
        if c % 100 == 0:
            logging.info(f'Iteration {c}')
        c += 1
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

        if c % 1000 == 0:
            try:
                logging.info(f'{c} -> Best result: {save_points[np.argmin(save_cvar)]}')
                logging.info(f'{c} -> CVaR: {save_cvar[np.argmin(save_cvar)]}')
            except Exception as e:
                logging.info(f'Error: {e}')

    with open(f'{OUTPUT_FOLDER}/rgs_output_{os.getenv("PBS_JOBID")}.pickle', 'wb') as f:
        pickle.dump({'points':save_points,
                    'cvar':save_cvar,
                    'constraint':save_con}, f)
        
    # print the best result
    best = np.argmin(save_cvar)
    logging.info(f'Best result: {save_points[best]}')
    logging.info(f'CVaR: {save_cvar[best]}')
