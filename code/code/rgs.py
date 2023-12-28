
#%%

import random
import numpy as np
from amm import amm
from params import params
from tqdm.auto import tqdm

# %% 

np.random.seed(params['seed']) #Fix the seed

#Initialize the pools
Rx0 = params['Rx0']
Ry0 = params['Ry0']
phi = params['phi']
pools = amm(Rx=Rx0, Ry=Ry0, phi=phi)

# Swap and mint
theta = [1/6]*6
xs_0 = params['x_0']*np.array(theta)
l = pools.swap_and_mint(xs_0)

# Simulate 100 paths of trading in the pools

end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t =\
    pools.simulate(
        kappa=params['kappa'],
        p=params['p'],
        sigma=params['sigma'],
        T=params['T'],
        batch_size=params['batch_size'])

print('Reserves in coin X for scenario 0:', np.round(end_pools[0].Rx, 2))
print('Reserves in coin Y for scenario 0:', np.round(end_pools[0].Ry, 2))

# Burns and swap all coins into x
x_T = np.zeros(batch_size)
for k in range(batch_size):
    x_T[k] = np.sum(end_pools[k].burn_and_swap(l))
x_0 = np.sum(xs_0)
log_ret = np.log(x_T) - np.log(x_0)
mean_ret = np.mean(log_ret)/T*100
print('Average performance :', round(mean_ret, 4))
print('Std. Dev. of performance:', round(np.std(log_ret)/np.sqrt(T)*100, 4))

#%% Plot the distribution of the final performance
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sns.set_theme()

fig, ax = plt.subplots(1, 1)

sns.histplot(ax = ax, x=log_ret, kde=True) #histogram plot
ax.set_xlabel('Return')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x*100)}%"))

# Compute cvar
alpha = 0.05
qtl = np.quantile(log_ret, alpha)
cvar = np.mean(log_ret[log_ret<=qtl])
plt.axvline(qtl, linestyle='--', label=rf'$VaR_\alpha = {round(qtl*100, 2)}\%$', color=sns.color_palette()[2])
plt.axvline(cvar, linestyle='--', label=rf'$CVaR_\alpha = {round(cvar*100, 2)}\%$', color=sns.color_palette()[1])
ax.legend()
plt.show()
