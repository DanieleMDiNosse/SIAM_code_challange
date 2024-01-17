
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
Summary of the best results
MINIMUM rgs
cvar = 

Top 5%
cvar = 

Top 1%
cvar = 

Equi-weighted
cvar = 0.005193897750906791


Mio IV
cvar = 0.004570863305932949

'''

OUTPUT_FOLDER = '/home/garo/Desktop/Lavoro_Studio/[SIAG] Challenge/SIAM_code_challange/code/temp_results'
'''
To obtain the percentile:
with open(f'{OUTPUT_FOLDER}/rgs_output.pickle', 'rb') as f:
    opt_res = pickle.load(f)
cvar_vals = np.array(opt_res['cvar'])
100*len(cvar_vals[cvar_vals <= cvar]) / len(cvar_vals)
'''

#%% Some useful function

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

    constraint= len(log_ret[ log_ret>params['zeta'] ]) / len(log_ret) - params['q']
    if constraint >= 0:
        cvar = calculate_cvar(log_ret) #If the constraint is satisfied, return CVaR
    else: #Otherwise, return np.inf or np.nan
        if ret_inf:
            cvar = np.nan
        else:
            cvar = np.inf
    return constraint, cvar

def pool_initializer(pars):
    #Initialize the pools
    Rx0 = pars['Rx0']
    Ry0 = pars['Ry0']
    phi = pars['phi']
    pools = amm(Rx=Rx0, Ry=Ry0, phi=phi)
    return pools

def market_simulator(pars, theta_, rho):
    # Eventually fill the theta vector
    if len(theta_) < params['N_pools']:
        theta = np.array([*theta_, 1-np.sum(np.abs(theta_))])
    else:
        theta = theta_

    np.random.seed(pars['seed']) #Fix the seed for the next operations
    pools = pool_initializer(pars) #Initialize the pools

    #The amount to invest on each pool is given by the weight in theta mult by the total capital
    xs_0 = pars['x_0']*theta*(1-rho)

    # We only consider a proportion (1-rho) of xs_0 when computing the quote
    try: #Adjustment for round-off errors
        l_fix = pools.swap_and_mint(xs_0)
    except AssertionError: #The AssertionError is when a x_0 component is too small (lees than 1e-6)
        print('Assertion error with rho = ', rho, 'and theta = ', theta)
        l_fix = np.zeros(len(theta))

    # Simulate 1000 paths of trading in the pools
    end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t =\
        pools.simulate(
            kappa=pars['kappa'],
            p=pars['p'],
            sigma=pars['sigma'],
            T=pars['T'],
            batch_size=pars['batch_size'])

    return end_pools#, l_fix, xs_0

def theta_filler(pars, theta, theta_adj, rho):
    xs_0 = pars['x_0']*theta*(1-rho) + pars['x_0']*theta_adj*rho

    pools = pool_initializer(pars) #Initialize the pools

    return pools.swap_and_mint(xs_0)

def approx2minimize(theta_adj_, pars, theta_, rho, end_pools):
    # Eventually fill the theta vector
    if len(theta_adj_) < params['N_pools']:
        theta_adj = np.array([*theta_adj_, 1-np.sum(np.abs(theta_adj_))])
    else:
        theta_adj = theta_adj_
    if len(theta_) < params['N_pools']:
        theta = np.array([*theta_, 1-np.sum(np.abs(theta_))])
    else:
        theta = theta_

    #theta_adj = np.array(theta_adj) / np.sum(theta_adj) #Regularize the weights
    l = theta_filler(pars, theta, theta_adj, rho) #Compute the quote

    # Update the coin avability
    for k in range(len(end_pools)):
        end_pools[k].l = l.copy()

    log_ret = calculate_log_returns(pars['x_0'], end_pools, l) #Compute the return
    
    return calculate_cvar(log_ret)

#%% First, easy version

'''
rho decreases exponentially
theta0 is equi-weighted
theta_approx is initialized to the current theta0
no early stopping strategy
no progressive increasing in the tolerance required to minimize
'''

from scipy.optimize import minimize, LinearConstraint

# Define algorithm hyperparameters
n_pools = 6
rho=0.9
theta0 = np.array([1/n_pools]*n_pools) 
max_rep = 6
min_seed = 2

#Set the constraint for the minimization algorithm
constr = LinearConstraint(
    np.concatenate([np.eye(n_pools), np.ones((1, n_pools))], axis=0),
    np.concatenate([np.ones(n_pools)*1e-5, [1]], axis=0),
    np.concatenate([np.ones(n_pools), [1]], axis=0),
    keep_feasible=True)

flag = True #Initialize the exit flag
rho_val = 1 #Initialize rho value

for _ in range(50):
    sim_pools = market_simulator(params, theta0, rho_val) #Simulate the market, starting point theta0

    # Find the optimal step with the interior minimization
    np.random.seed(min_seed)
    theta = theta0
    temp_flag, it = 1, 0

    while (temp_flag!=0) and (it < max_rep):
        res = minimize(
            lambda x: approx2minimize(x, params, theta0,
                                      rho_val, copy.deepcopy(sim_pools)),
            theta, method='SLSQP', constraints=constr, options={'disp':False}
            ) #Minimize to find the next theta step
        theta = np.array(res.x)
        temp_flag = res.status
        #print(theta)
        #print(temp_flag)
        it += 1

    print(temp_flag)
    print(rho_val)

    # Update variables
    theta0 = theta0*(1-rho_val) + theta*rho_val
    rho_val *= rho
    print(theta0)
    cons, cvar = target_4_opt(theta0, params)
    print(cvar)
    print(cons)

'''
theta0 = 
cvar = 
'''

#%% Second version

'''
rho decreases exponentially
theta0 is 1/sigma
theta_approx is initialized to the current theta0
no early stopping strategy
no progressive increasing in the tolerance required to minimize
'''

from scipy.optimize import minimize, LinearConstraint

# Define algorithm hyperparameters
n_pools = 6
rho=0.9
theta0 = 1/(np.array(params['sigma'][1:]))
theta0 = theta0/np.sum(theta0)
max_rep = 6
min_seed = 2

#Set the constraint for the minimization algorithm
constr = LinearConstraint(
    np.concatenate([np.eye(n_pools), np.ones((1, n_pools))], axis=0),
    np.concatenate([np.ones(n_pools)*1e-5, [1]], axis=0),
    np.concatenate([np.ones(n_pools), [1]], axis=0),
    keep_feasible=True)

flag = True #Initialize the exit flag
rho_val = 1 #Initialize rho value

for _ in range(50):
    sim_pools = market_simulator(params, theta0, rho_val) #Simulate the market, starting point theta0

    # Find the optimal step with the interior minimization
    np.random.seed(min_seed)
    theta = theta0
    temp_flag, it = 1, 0

    while (temp_flag!=0) and (it < max_rep):
        res = minimize(
            lambda x: approx2minimize(x, params, theta0,
                                      rho_val, copy.deepcopy(sim_pools)),
            theta, method='SLSQP', constraints=constr, options={'disp':False}
            ) #Minimize to find the next theta step
        theta = np.array(res.x)
        temp_flag = res.status
        #print(theta)
        #print(temp_flag)
        it += 1

    print(temp_flag)
    print(rho_val)

    # Update variables
    theta0 = theta0*(1-rho_val) + theta*rho_val
    rho_val *= rho
    print(theta0)
    cons, cvar = target_4_opt(theta0, params)
    print(cvar)
    print(cons)

'''
theta0 = 
cvar = 
'''

#%% Third version

'''
rho decreases exponentially
theta0 is equi-weighted
theta_approx is initialized to equi-weighted
no early stopping strategy
no progressive increasing in the tolerance required to minimize
'''

from scipy.optimize import minimize, LinearConstraint

# Define algorithm hyperparameters
n_pools = 6
rho=0.9
theta0 = np.array([1/n_pools]*n_pools) 
max_rep = 6
min_seed = 2

#Set the constraint for the minimization algorithm
constr = LinearConstraint(
    np.concatenate([np.eye(n_pools), np.ones((1, n_pools))], axis=0),
    np.concatenate([np.ones(n_pools)*1e-5, [1]], axis=0),
    np.concatenate([np.ones(n_pools), [1]], axis=0),
    keep_feasible=True)

flag = True #Initialize the exit flag
rho_val = 1 #Initialize rho value

for g_it in range(50):
    sim_pools = market_simulator(params, theta0, rho_val) #Simulate the market, starting point theta0

    # Find the optimal step with the interior minimization
    np.random.seed(min_seed)
    theta = np.array([1/n_pools]*n_pools)
    temp_flag, it = 1, 0

    while (temp_flag!=0) and (it < max_rep):
        res = minimize(
            lambda x: approx2minimize(x, params, theta0,
                                      rho_val, copy.deepcopy(sim_pools)),
            theta, method='SLSQP', constraints=constr, tol=1e-12, options={'disp':False}
            ) #Minimize to find the next theta step
        theta = np.array(res.x)
        temp_flag = res.status
        #print(theta)
        #print(temp_flag)
        it += 1

    print(temp_flag)
    print(rho_val)

    # Update variables
    theta0 = theta0*(1-rho_val) + theta*rho_val
    rho_val *= rho
    print(theta0)
    if g_it % 10 == 0:
        cons, cvar = target_4_opt(theta0, params)
        print(cvar)
        print(cons)

'''
theta0 = [0.17736765, 0.15817278, 0.14903491, 0.18351665, 0.22786052, 0.10404748]
cvar = 0.030728284345139727
'''

#%% Fourth

'''
rho decreases exponentially; rho=0.7
theta0 is equi-weighted
theta_approx is initialized to the current theta0
no early stopping strategy
no progressive increasing in the tolerance required to minimize
'''

from scipy.optimize import minimize, LinearConstraint

# Define algorithm hyperparameters
n_pools = len(params['Rx0'])
rho=0.7
theta0 = np.array([1/n_pools]*n_pools) 
max_rep = 6
min_seed = 2
int_tol = 1e-12
ext_tol = 1e-6
max_cte = 2

#Set the constraint for the minimization algorithm
constr = LinearConstraint(
    np.concatenate([np.eye(n_pools), np.ones((1, n_pools))], axis=0),
    np.concatenate([np.ones(n_pools)*1e-5, [1]], axis=0),
    np.concatenate([np.ones(n_pools), [1]], axis=0),
    keep_feasible=True)

flag = True #Initialize the exit flag
rho_val = 1 #Initialize rho value
Thetas_record = list() #Store theta for future analysis
exit_count = 0
theta0_old = theta0

for _ in range(20):
    sim_pools = market_simulator(params, theta0, rho_val) #Simulate the market, starting point theta0

    # Find the optimal step with the interior minimization
    np.random.seed(min_seed)
    theta = theta0
    temp_flag, it = 1, 0

    while (temp_flag!=0) and (it < max_rep):
        res = minimize(
            lambda x: approx2minimize(x, params, theta0,
                                      rho_val, copy.deepcopy(sim_pools)),
            theta, method='SLSQP', constraints=constr, tol=int_tol, options={'disp':False}
            ) #Minimize to find the next theta step
        theta = np.array(res.x)
        temp_flag = res.status
        #print(theta)
        #print(temp_flag)
        it += 1

    print(temp_flag)
    print(rho_val)

    # Update variables
    theta0 = theta0*(1-rho_val) + theta*rho_val
    rho_val *= rho
    Thetas_record.append(theta0)
    print(theta0)
    # Check the exit condition
    if np.sum(np.abs(theta0-theta0_old)) <= ext_tol:
        exit_count += 1
        if exit_count >= max_cte:
            break
    else:
        exit_count = 0
    theta0_old = theta0

'''
theta0 = np.array([0.17415879, 0.16584002, 0.1529393 , 0.18366579, 0.22503606, 0.09836005])
cvar = 0.004570863305932949
'''

#%% V version

'''
rho decreases exponentially; rho=0.8
theta0 is equi-weighted
theta_approx is initialized to the current theta0
no early stopping strategy
yes progressive increasing in the tolerance required to minimize
'''

from scipy.optimize import minimize, LinearConstraint

# Define algorithm hyperparameters
n_pools = 6
rho=0.8
theta0 = np.array([1/n_pools]*n_pools)
max_rep = 6
min_seed = 2
init_tol = 1e-5

#Set the constraint for the minimization algorithm
constr = LinearConstraint(
    np.concatenate([np.eye(n_pools), np.ones((1, n_pools))], axis=0),
    np.concatenate([np.ones(n_pools)*1e-5, [1]], axis=0),
    np.concatenate([np.ones(n_pools), [1]], axis=0),
    keep_feasible=True)

flag = True #Initialize the exit flag
rho_val = 1 #Initialize rho value
Thetas_record = list() #Store theta for future analysis

for g_it in range(30):
    sim_pools = market_simulator(params, theta0, rho_val) #Simulate the market, starting point theta0

    # Find the optimal step with the interior minimization
    np.random.seed(min_seed)
    theta = theta0
    t_flag, step_tol = True, 1

    while t_flag and (step_tol>=1e-8):
        res = minimize(
            lambda x: approx2minimize(x, params, theta0,
                                      rho_val, copy.deepcopy(sim_pools)),
            theta, method='SLSQP', tol=init_tol*step_tol,
            constraints=constr, options={'disp':False}
            ) #Minimize to find the next theta step
        theta = np.array(res.x)
        temp_flag = res.status
        if np.abs(theta-theta0).sum() > 1e-7:
            t_flag = False
        else:
            step_tol *= 0.1



        # print(theta)
        # print(temp_flag)
        # print(np.abs(theta-theta0))



    print(temp_flag)
    print(rho_val)

    # Update variables
    theta0 = theta0*(1-rho_val) + theta*rho_val
    rho_val *= rho
    Thetas_record.append(theta0)
    print(theta0)
    if g_it < 4:
        cons, cvar = target_4_opt(theta0, params)
        print(cvar)
        print(cons)
    elif g_it % 10 == 0:
        cons, cvar = target_4_opt(theta0, params)
        print(cvar)
        print(cons)

'''
theta0 = 
cvar = 
'''

#%% VI version

'''
rgs minimizer
rho decreases exponentially; rho=0.7
theta0 is equi-weighted
theta_approx is initialized to the current theta0
no early stopping strategy
no progressive increasing in the tolerance required to minimize
'''

from scipy.optimize import minimize, LinearConstraint

# Define algorithm hyperparameters
n_pools = 6
rho=0.7
theta0 = np.array([1/n_pools]*n_pools) 
max_rep = 6
min_seed = 2

#Set the constraint for the minimization algorithm
constr = LinearConstraint(
    np.concatenate([np.eye(n_pools), np.ones((1, n_pools))], axis=0),
    np.concatenate([np.ones(n_pools)*1e-5, [1]], axis=0),
    np.concatenate([np.ones(n_pools), [1]], axis=0),
    keep_feasible=True)

flag = True #Initialize the exit flag
rho_val = 1 #Initialize rho value
Thetas_record = list() #Store theta for future analysis

for _ in range(20):
    sim_pools = market_simulator(params, theta0, rho_val) #Simulate the market, starting point theta0

    # Find the optimal step with the interior minimization
    best_val, theta = np.inf, None
    for rgs_seed in tqdm(range(400)):
        np.random.seed(rgs_seed)
        #Randomly draw the vector of weights theta and refularize it to have sum=1
        temp_theta = list()
        for _ in range(len(params['Rx0'])):
            temp_theta.append( np.random.uniform() )
        temp_theta = np.array(temp_theta) / np.sum(temp_theta)
        val = approx2minimize(temp_theta, params, theta0, rho_val, copy.deepcopy(sim_pools))
        if val < best_val:
            best_val = val
            theta = temp_theta

    print(rho_val)
    # Update variables
    theta0 = theta0*(1-rho_val) + theta*rho_val
    rho_val *= rho
    Thetas_record.append(theta0)
    print(theta0)

'''
theta0 = 
cvar = 
'''

#%% VII version

'''
rho decreases exponentially; rho=0.7
theta0 is equi-weighted
theta_approx is initialized to the current theta0
no early stopping strategy
no progressive increasing in the tolerance required to minimize
'''

from scipy.optimize import minimize, LinearConstraint

# Define algorithm hyperparameters
n_pools = len(params['Rx0'])
rho=0.7
theta0 = np.array([1/n_pools]*(n_pools-1)) 
max_rep = 6
min_seed = 2
int_tol = 1e-12
ext_tol = 1e-6
max_cte = 2

#Set the constraint for the minimization algorithm
constr = LinearConstraint(
    np.concatenate([np.eye(n_pools), np.ones((1, n_pools))], axis=0),
    np.concatenate([np.ones(n_pools)*1e-5, [1]], axis=0),
    np.concatenate([np.ones(n_pools), [1]], axis=0),
    keep_feasible=True)

constraints = [{'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)}]
bounds = [(0, 1) for i in range(params['N_pools']-1)]

flag = True #Initialize the exit flag
rho_val = 1 #Initialize rho value
Thetas_record = list() #Store theta for future analysis
exit_count = 0
theta0_old = theta0

for _ in range(20):
    sim_pools = market_simulator(params, theta0, rho_val) #Simulate the market, starting point theta0

    # Find the optimal step with the interior minimization
    np.random.seed(min_seed)
    theta = theta0
    temp_flag, it = 1, 0

    while (temp_flag!=0) and (it < max_rep):
        res = minimize(
            lambda x: approx2minimize(x, params, theta0,
                                      rho_val, copy.deepcopy(sim_pools)),
            theta, method='trust-constr', constraints=constraints,
            bounds=bounds, tol=int_tol, options={'disp':False}
            ) #Minimize to find the next theta step
        theta = np.array(res.x)
        temp_flag = res.status
        #print(theta)
        #print(temp_flag)
        it += 1

    print(temp_flag)
    print(rho_val)

    # Update variables
    theta0 = theta0*(1-rho_val) + theta*rho_val
    rho_val *= rho
    Thetas_record.append(theta0)
    print(theta0)
    if np.sum(np.abs(theta0-theta0_old)) <= ext_tol:
        exit_count += 1
        if exit_count >= max_cte:
            break
    else:
        exit_count = 0
    theta0_old = theta0

'''
theta0 = np.array([])
cvar = 
'''

# %% SPACE


#%% Where is the problem? Compare minimize and rgs with rho=1 and theta0 equi-weighted

theta0 = np.array([1/n_pools]*n_pools) 
rho = 1

best_val, best_theta = np.inf, None
sim_pools = market_simulator(params, theta0, rho_val)
for rgs_seed in tqdm(range(400)):
    np.random.seed(rgs_seed)

    #Randomly draw the vector of weights theta and refularize it to have sum=1
    temp_theta = list()
    for _ in range(len(params['Rx0'])):
        temp_theta.append( np.random.uniform() )
    temp_theta = np.array(temp_theta) / np.sum(temp_theta)

    val = approx2minimize(temp_theta, params, theta0, rho_val, copy.deepcopy(sim_pools))
    #print(val)
    if val < best_val:
        best_val = val
        best_theta = temp_theta

print(best_theta)
sim_pools = market_simulator(params, theta0, rho_val)
print(approx2minimize(best_theta, params, theta0, rho_val, copy.deepcopy(sim_pools)))

#------------------------------------------

sim_pools = market_simulator(params, theta0, rho_val) #Simulate the market, starting point theta0

# Minimization
np.random.seed(min_seed)

res = minimize(
    lambda x: approx2minimize(x, params, theta0, rho_val,
                              copy.deepcopy(sim_pools)),
    theta0, method='SLSQP', constraints=constr, tol=1e-20, options={'disp':False}
    ) #Minimize to find the next theta step
theta = np.array(res.x)

print(theta)
sim_pools = market_simulator(params, theta0, rho_val)
print(approx2minimize(theta, params, theta0, rho_val, copy.deepcopy(sim_pools)))

# %%
