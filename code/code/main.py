import numpy as np
import matplotlib.pyplot as plt
from utils import *
from amm import amm
from utils_class import PortfolioOptimizer
from params import params
import argparse
import datetime

parser = argparse.ArgumentParser(description='Optimize the initial wealth distribution')
parser.add_argument('-s', '--simulate', type=int, default=0, help='Simulate the model with the optimal initial wealth distribution (1) or not (0)')
parser.add_argument('-m', '--method', type=str, default='SLSQP', help='Optimization method for scipy.minimize')
args = parser.parse_args()

_ = logging_config('opt')
get_current_git_branch()
np.random.seed(params['seed'])
res = optimize_distribution(params, args.method)

# optimizer = PortfolioOptimizer()
# res = optimizer.optimize_distribution(params, args.method)
logging.info(f'Time: {datetime.datetime.now()}')

if args.simulate == 1:
#     res = np.array([2.48080213e-01, 2.02686541e-01, 2.00305892e-01, 2.11960272e-01,
#  1.36966020e-01, 1.07737608e-06])
    simulation_plots(res, params)
    # optimizer.simulation_plots(res, params)