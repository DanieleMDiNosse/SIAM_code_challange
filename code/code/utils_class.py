import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy
import logging
import os

from amm import amm
from params import params

class PortfolioOptimizer:
    def __init__(self):
        self.log_returns = 0
        self.probability = 0
        self.cvar = 0
        self.lambda1 = 20
        self.lambda2 = 10
        self.x = np.zeros(params['N_pools'])

    @staticmethod
    def calculate_cvar(log_returns, x, lambda1, lambda2, probability):
        """
        Calculate the CVaR of a set of returns.
        """

        var = np.quantile(-log_returns, params['alpha'])
        cvar = np.mean(-log_returns[-log_returns >= var])
        cvar_w_penalty = cvar + lambda1 * (np.sum(np.abs(x)) - 1) + lambda2 * max(0, (params['q'] - probability)) + lambda2 * np.sum(np.maximum(0, -x))
        return cvar_w_penalty, cvar, var

    @staticmethod
    def calculate_log_returns(x0, final_pools_dists, l):
        """
        In the simulate method of the amm class, a  certain number of trading paths are
        generated. Consequently, for each path we have a final log return.
        This function calculates the log returns for each final state of the pools over
        all the simulated paths.
        """
        x_T = np.zeros(params['batch_size']) # each element of x_T will be the final wealth of a path

        # In order to have the final wealth, we need to burn and swap the LP tokens
        # for each path. This is done by the burn_and_swap method of the amm class.
        # The method takes all the LP tokens, burn them and swap coin-Y for coin-X.
        for k in range(params['batch_size']):
            x_T[k] = np.sum(final_pools_dists[k].burn_and_swap(l))

        # Calculate the initial wealth
        x_0 = np.sum(x0)

        # Calculate the log returns for each path
        log_returns = np.log(x_T) - np.log(x_0)

        return log_returns


    def portfolio_evolution(self, initial_pools_dist, amm_instance_, params):
        # Avoid the modification of the amm instance every function call
        self.x = initial_pools_dist
        amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])
        # Check if there is a negative weight
        if np.any(initial_pools_dist < 0):
            logging.info(f'Negative weight: {initial_pools_dist}')
            initial_pools_dist = np.abs(initial_pools_dist)

        # Compute the actual tokens for each pool
        X0 = params['x_0'] * initial_pools_dist

        # Evaluate the number of LP tokens for returns computation
        try:
            l = amm_instance.swap_and_mint(X0)
        except AssertionError as e:
            logging.info(f"Error: {e}")
            return 1e6

        # Simulate the evolution of the pools (scenario simulation)
        np.random.seed(params['seed'])

        final_pools_dists, _, _, _, _, _ = amm_instance.simulate(
            kappa=params['kappa'], p=params['p'], sigma=params['sigma'], 
            T=params['T'], batch_size=params['batch_size'])

        # Calculate the log returns for each path
        self.log_returns = self.calculate_log_returns(X0, final_pools_dists, l)

        # Compute the cvar
        cvar_w_penalty, self.cvar, _ = self.calculate_cvar(self.log_returns, self.x, self.lambda1, self.lambda2, self.probability)

        # Compute the probability of having a return greater than 0.05
        self.probability = np.mean(self.log_returns > 0.05)

        return cvar_w_penalty

    def optimize_distribution(self, params, method):
        # ... (similar to before, but replace global variables with self.attributes)
        # ... (use self.portfolio_evolution, self.constraint_1, and self.constraint_2)
        """
        Optimizes the distribution of wealth across liquidity pools to minimize CVaR,
        conditioned to P[final return > 0.05]>params['q'].

        Args:
        - amm_instance (amm): Instance of the amm class.
        - params (dict): Parameters for the amm and optimization.

        Returns:
        - dict: Optimal weights and corresponding CVaR.
        """
        np.random.seed(params['seed'])

        # Bounds
        bounds_initial_dist = [(1e-5, 1) for i in range(params['N_pools'])]

        # Instantiate the amm class
        amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])

        # Callback function to print the current CVaR and the current parameters
        def callback_function(x, *args):
            # current_cvar, _ = calculate_cvar(log_returns)
            logging.info(f"Current initial_dist: {x} -> Sum: {np.sum(x)}")
            logging.info(f"Current probability: {self.probability}")
            logging.info(f'Mean loss: {np.mean(-self.log_returns)}')
            logging.info(f"Current VaR:{np.quantile(-self.log_returns, params['alpha'])}")
            logging.info(f"Current CVaR: {self.cvar}\n")

        # The following while loop is used to check if the initial distribution of wealth
        # across pools is feasible. If it is not, a new one is generated.
        cond = True
        while cond == True:
            # Initial distribution of wealth across pools
            random_vector = np.random.uniform(0, 100, params['N_pools'])
            initial_guess = random_vector / np.sum(random_vector)
            self.x = initial_guess
            try:
                amm_instance.swap_and_mint(initial_guess*params['x_0'], quote=True)
                cond = False
            except ValueError as e:
                logging.info(f"Error: {e}")
        logging.info(f"Initial guess:\n\t{initial_guess}\n")

        # Optimization procedure
        logging.info(f"Minimization of vanilla cVaR with penalties")
        logging.info(f"Optimization method: {method}")
        logging.info("Starting...")
        logging.info(f"batch size:\n\t{params['batch_size']}\n")
        result = minimize(self.portfolio_evolution, initial_guess, args=(amm_instance, params),
                    method=method, bounds=bounds_initial_dist, callback=callback_function)

        logging.info(f"Results:\n\t{result}")

        return result.x
    
    def simulation_plots(self, res, params):
        """
        Plots the evolution of the reserves, the price, and the returns for a given
        initial distribution of wealth across pools.
        """

        X0 = res[-params['N_pools']:] * params['x_0']

        # Initialize the pools
        amm_instance = amm(params['Rx0'], params['Ry0'], params['phi'])

        # Evaluate the number of LP tokens
        l = amm_instance.swap_and_mint(X0)

        # Simulate the evolution of the pools
        np.random.seed(params['seed'])

        XT, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = amm_instance.simulate(
            kappa=params['kappa'], p=params['p'], sigma=params['sigma'], 
            T=params['T'], batch_size=params['batch_size'])

        # Calculate the log returns, cvar, and var
        log_returns = self.calculate_log_returns(X0, XT, l)
        probability = np.mean(log_returns > 0.05)
        cvar_w_penalty, cvar, var = self.calculate_cvar(log_returns, res, self.lambda1, self.lambda2, probability)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        i = np.random.randint(0, params['N_pools'])
        
        # Plot the evolution of the reserves
        ax[0].plot(Rx_t[i])
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('X-Reserves')

        ax[1].plot(Ry_t[i])
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Y-Reserves')

        # Plot the evolution of the marginal price
        ax[2].plot(np.array(Rx_t[i])/np.array(Ry_t[i]))
        ax[2].set_xlabel('Time')
        ax[2].set_ylabel('Marginal Price')
        plt.savefig(f'pools_{os.getenv("PBS_JOBID")}.png')

        # Plot the distribution of the returns
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.hist(log_returns, bins=50, alpha=0.7)
        plt.axvline(-cvar, color='r', linestyle='dashed', linewidth=1, label='CVaR')
        plt.axvline(-var, color='b', linestyle='dashed', linewidth=1, label='VaR')
        plt.axvline(0.05, color='g', linestyle='dashed', linewidth=1, label=r'$\xi$', alpha=0.0)
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend([f'CVaR:{-cvar}', f'VaR:{-var}', fr'$E[r_T]$:{np.mean(log_returns):.3f}', fr'$P[r_T>\xi]$:{probability:.3f}'])
        plt.savefig(f'returns_{os.getenv("PBS_JOBID")}.png')

        plt.show()
