import numpy as np
import matplotlib.pyplot as plt
from utils import *
from amm import amm
from params import params
import datetime
import argparse

parser = argparse.ArgumentParser(description='Optimize the initial wealth distribution')
parser.add_argument('-s', '--simulate', type=int, default=0, help='Simulate the model with the optimal initial wealth distribution (1) or not (0)')
parser.add_argument('-m', '--method', type=str, default='SLSQP', help='Optimization method for scipy.minimize')
parser.add_argument('-uc', '--unconstraint', action='store_true', help='Choose between constrained or unconstrained optimization')
args = parser.parse_args()

_ = logging_config('opt')
get_current_git_branch()
np.random.seed(params['seed'])
res = optimize_distribution(params, args.method, args.unconstraint)
logging.info(f'Time: {datetime.datetime.now()}')

if args.simulate == 1:
    simulation_plots(res, params)