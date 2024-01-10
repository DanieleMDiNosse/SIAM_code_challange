
#%% Importing

import copy
import pickle
import random
import numpy as np
from amm import amm
import seaborn as sns
from params import params
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils import calculate_cvar, calculate_log_returns, target_4_opt
sns.set_theme()

OUTPUT_FOLDER = '/home/garo/Desktop/Lavoro_Studio/[SIAG] Challenge/SIAM_code_challange/code/temp_results'

# %% Some plots to have an idea...

#Load the data
with open(f'{OUTPUT_FOLDER}/rgs_output.pickle', 'rb') as f:
    opt_res = pickle.load(f)
    
print(f'There are {100*np.isnan(opt_res["cvar"]).sum()/len(opt_res["cvar"])}% NaN')
# Plot the cvar as a function of the constraint. I'm trying to understand
#   whenever the constraint is active (i.e. cvar minimum is on the boundary con=0)

fig, ax = plt.subplots(1, 1)
sns.scatterplot(x=np.array(opt_res['constraint'])[ ~ np.isnan(opt_res['cvar']) ],
                y=np.array(opt_res['cvar'])[ ~ np.isnan(opt_res['cvar']) ], ax = ax)
ax.axhline(0.03121627321033861, color=sns.color_palette()[1], linestyle='--', label='Equi-weighted portfolio')
ax.set_xlabel(r'Constraint $\mathbb{P}[r_T > \zeta] - 0.7$')
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

#%% Is the loss function convex?

flag = False
for rgs_seed in tqdm(range(581, 1000), desc='Random Grid Search'):
    np.random.seed(rgs_seed)

    #Randomly draw the vector of weights theta and refularize it to have sum=1
    theta_1, theta_2 = list(), list()
    for _ in range(len(params['Rx0'])):
        theta_1.append( np.random.uniform() )
        theta_2.append( np.random.uniform() )
    theta_1 = np.array(theta_1) / np.sum(theta_1)
    _, cvar1 = target_4_opt(theta_1, params, ret_inf=False)
    theta_2 = np.array(theta_2) / np.sum(theta_2)
    _, cvar2 = target_4_opt(theta_2, params, ret_inf=False)

    for lam in np.arange(0.1, 0.9, 0.25):
        _, cvar0 = target_4_opt(
            lam*theta_1 + (1-lam)*theta_2, params, ret_inf=False)
        if cvar0 > lam*cvar1 + (1-lam)*cvar2:
            print(theta_1)
            print(theta_2)
            print(lam)
            flag = True
            break
    if flag:
        break

'''
Critical points:
rgs_seed = 444
theta_1 = [0.2955491, 0.22612885, 0.01378163, 0.09735919, 0.06816621, 0.29901502]
theta_2 = [0.28223181, 0.22928601, 0.02256782, 0.1087149, 0.06772948, 0.28946999]
lam = 0.35; lam = 0.6; lam = 0.85; lam = 0.5

rgs_seed = 574
theta_1 = [0.16773388, 0.15768413, 0.12888258, 0.21354976, 0.05863106, 0.2735186]
theta_2 = [0.27333687, 0.00074449, 0.09398424, 0.20518744, 0.13048686, 0.29626009]
lam = 0.35

rgs_seed = 580
theta_1 = [0.2649939, 0.03223408, 0.18399521, 0.1883413, 0.15597629, 0.17445922]
theta_2 = [0.12810762, 0.20746266, 0.20485631, 0.19563079, 0.12317922, 0.1407634]
lam = 0.85
'''

#%% Is the approximated loss function convex?

flag = False
for rgs_seed in tqdm(range(1000), desc='Random Grid Search'):
    np.random.seed(rgs_seed)

    # Randomly draw the vector of weights theta and refularize it to have sum=1
    rho = np.random.uniform()
    theta0 = list()
    for _ in range(len(params['Rx0'])):
        theta0.append( np.random.uniform() )
    theta_1, theta_2 = list(), list()
    for _ in range(len(params['Rx0'])):
        theta_1.append( np.random.uniform() )
        theta_2.append( np.random.uniform() )

    # Randomly simulate the market scenario
    theta0 = np.array(theta0) / np.sum(theta0)
    sim_pools = market_simulator(params, theta0, rho)
    theta_1 = np.array(theta_1) / np.sum(theta_1)
    cvar1 = approx2minimize(theta_1, params, theta0, rho, copy.deepcopy(sim_pools))
    theta_2 = np.array(theta_2) / np.sum(theta_2)
    cvar2 = approx2minimize(theta_2, params, theta0, rho, copy.deepcopy(sim_pools))

    for lam in np.arange(0.1, 0.9, 0.25):
        cvar0 = approx2minimize(lam*theta_1 + (1-lam)*theta_2,
                                params, theta0, rho, copy.deepcopy(sim_pools))
        if cvar0 > lam*cvar1 + (1-lam)*cvar2:
            print(rgs_seed)
            print(theta_1)
            print(theta_2)
            print(lam)
            flag = True
            break
    if flag:
        break
    