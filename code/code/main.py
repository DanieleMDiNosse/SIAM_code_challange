import numpy as np
import matplotlib.pyplot as plt
from utils import *
from amm import amm
from params import params
import argparse

parser = argparse.ArgumentParser(description='Optimize the initial wealth distribution')
parser.add_argument('-s', '--simulate', type=int, default=0, help='Simulate the model with the optimal initial wealth distribution (1) or not (0)')
parser.add_argument('-m', '--method', type=str, default='SLSQP', help='Optimization method for scipy.minimize')
args = parser.parse_args()

_ = logging_config('opt')
np.random.seed(params['seed'])
res = optimize_distribution(params, args.method)

if args.simulate == 1:
    # res = np.array([0.11057456, 0.34389733, 0.17021943, 0.13990844, 0.22973344,
    #    0.0056668 ])
    simulation_plots(res, params)