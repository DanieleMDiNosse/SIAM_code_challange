
#%% Importing

import copy
import pickle
import random
import numpy as np
from amm import amm
from params import params
from tqdm.auto import tqdm
from utils import calculate_cvar, calculate_log_returns

'''
MINIMUM approx
[0.11057456, 0.34389733, 0.17021943, 0.13990844, 0.22973344, 0.0056668]
cvar = 0.029587175946245044

Top 5%
cvar = 0.030439706303845397

Top 1%
cvar = 0.03004233448060176

Equi-weighted
cvar = 0.03121627321033861
cvar_vals = np.array(opt_res['cvar'])
100*len(cvar_vals[cvar_vals <= cvar]) / len(cvar_vals)
21.78%

Mio Easy
cvar = 0.030738017656835127
10.22%
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

    constraint= len(log_ret[ log_ret>params['zeta'] ]) / len(log_ret) - 0.7
    if constraint >= 0:
        cvar = calculate_cvar(log_ret) #If the constraint is satisfied, return CVaR
    else: #Otherwise, return np.inf or np.nan
        if ret_inf:
            cvar = np.nan
        else:
            cvar = np.inf
    return constraint, cvar

#%% Random Grid Search

save_points = list() #Initialize lists of result
save_cvar = list() 
save_con = list()

# Iterate over the random seed
for rgs_seed in tqdm(range(12000), desc='Random Grid Search'):
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

with open(f'{OUTPUT_FOLDER}/rgs_output.pickle', 'wb') as f:
    pickle.dump({'points':save_points,
                 'cvar':save_cvar,
                 'constraint':save_con}, f)
