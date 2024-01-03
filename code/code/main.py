import numpy as np
import matplotlib.pyplot as plt
from utils import *
from amm import amm
from params import params
import argparse

parser = argparse.ArgumentParser(description='Optimize the initial wealth distribution')
parser.add_argument('-s', '--simulate', type=int, default=0, help='Simulate the model with the optimal initial wealth distribution (1) or not (0)')
args = parser.parse_args()

logging_config()
np.random.seed(params['seed'])
res = optimize_distribution(params)

if args.simulate == 1:
    simulation_plots(res, params)