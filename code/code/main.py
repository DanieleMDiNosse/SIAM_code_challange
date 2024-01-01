import numpy as np
import matplotlib.pyplot as plt
from utils import *
from amm import amm
from params import params
import argparse

parser = argparse.ArgumentParser(description='Optimize the initial wealth distribution')
parser.add_argument('-s', '--simulate', type=int, default=0, help='Simulate the model with the optimal initial wealth distribution (1) or not (0)')
args = parser.parse_args()

np.random.seed(params['seed'])
optimal_x0 = optimize_distribution(params)

if args.simulate == 1:
    simulation_plots(optimal_x0[-params['N_pools']:], params)