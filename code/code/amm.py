"""
Created on Fri Nov  21 17:26:27 2023

@author: jaimungal
"""

import numpy as np
from collections import deque
import copy
import time

from tqdm import tqdm

class amm():

    def __init__(self, Rx, Ry,  phi):
        """
        instantiate the class

        Parameters
        ----------
        Rx : array (K,)
            initial reservese of token-X in each pool
        Ry : array (K,)
            initial reservese of token-Y in each pool
        phi : array (K,)
            the pool fee

        Returns
        -------
        None.

        """
        
        assert (len(Rx) == len(Ry)) & (len(Ry)==len(phi)), "length of Rx, Ry, and phi must be the same."

        self.Rx = 1*Rx
        self.Ry = 1*Ry
        self.phi = 1*phi
        self.N = len(self.Rx)

        # number of LP tokens for each pool
        self.L = np.sqrt(self.Rx*self.Ry)

        # the trader begins with no LP tokens
        self.l = np.zeros(len(self.L))

    def swap_x_to_y(self, x, quote=False):
        """
        swap token-X for token-Y across all pools simulataneously

        Parameters
        ----------
        x : array (K,)
            the amount of token-X to swap in each AMM pool.
        quote: bool, optional
            deafult is False.
            If False, then pool states are updated.
            If True, pool states are not updated.

        Returns
        -------
        y : array (K,)
            the amount of token-Y you receive from each pool.

        """

        # Compute the amount of token-Y you receive from each pool
        y = x*self.Ry*(1-self.phi) / (self.Rx + x*(1-self.phi))

        # If quote is False, then update the pool states
        if quote is False:
            self.Rx += x
            self.Ry -= y

        return y

    def swap_y_to_x(self, y, quote=False):
        """
        swap token-Y for token-X across all pools simulataneously

        Parameters
        ----------
        y : array (K,)
            the amount of token-Y to swap in each AMM pool.
        quote: bool, optional
            deafult is False.
            If False, then pool states are updated.
            If True, pool states are not updated.

        Returns
        -------
        x : array (K,)
            the amount of token-X you receive from each pool.

        """

        # Compute the amount of token-X you receive from each pool
        x = y*self.Rx*(1-self.phi) / (self.Ry + y*(1-self.phi))

        # If quote is False, then update the pool states
        if quote is False:
            self.Rx -= x
            self.Ry += y

        return x

    def mint(self, x, y):
        """
        mint LP tokens across all pools

        Parameters
        ----------
        x : array (K,)
            amount of token-X submitted to each pool.
        y : array (K,)
            amount of token-Y submitted to each pool.

        Returns
        -------
        l : array (K,)
            The amount of LP tokens you receive from each pool.

        """

        for k in range(len(self.Rx)):
            assert np.abs(((x[k]/y[k])-self.Rx[k]/self.Ry[k])) < 1e-9, "pool " + str(k) + f" has incorrect submission of tokens:\tSubmission ratio: {x[k]/y[k]:.9f}\tMarginal Price: {self.Rx[k]/self.Ry[k]:.9f}"

        # Compute the amount of LP tokens you receive from each pool
        l = x*self.L/self.Rx
        for i in range(len(l)):
            if l[i] < 0:
                print(x[i], self.L[i], self.Rx[i])
            
        # Update the pool states
        self.Rx += x
        self.Ry += y
        self.L += l
        self.l += l

        return l

    def swap_and_mint(self, x, quote=False):
        """
        a method that determines the correct amount of y for each x within the corresponding pool
        to swap and then mint tokens with the reamaing x and the y you received

        Parameters
        ----------
        x : array (K,)
            amount of token-X you have for each pool.

        Returns
        -------
        l : array (K,)
            The amount of LP tokens you receive from each pool.

        """
        
        # Compute the percentage to swap in each pool
        x = np.array(x, float)
        theta = 1 + (2 - self.phi)*self.Rx*(1 - np.sqrt(
            1+4*x*(1-self.phi)/(self.Rx*(2-self.phi)**2)
            )) / ( 2*(1 - self.phi)*x )

        return self.mint(x*theta, self.swap_x_to_y(x*(1-theta), quote=False))
    
    def burn_and_swap(self, l):
        """
        a method that burns your LP tokens, then swaps y to x and returns only x

        Parameters
        ----------
        l : array (K,)
            amount of LP tokens you have for each pool.

        Returns
        -------
        x : array (K,)
            The amount of token-x you receive at the end.

        """
        
        x, y = self.burn(l)
        # Request quote
        quote = self.swap_y_to_x([y.sum()]*self.N, quote=True)
        # Assign y to the pool with the highest quote
        to_swap = np.zeros_like(quote)
        to_swap[np.argmax(quote)] = y.sum()
        # Swap coin
        total_x = (x + self.swap_y_to_x(to_swap, quote=False)).sum() #Swap coin
        return total_x

    def burn(self, l):
        """
        burn LP tokens across all pools

        Parameters
        ----------
        l : array (K,)
            amount of LP tokens to burn

        Returns
        -------
        x : array (K,)
            The amount of token-X received across
        y : array (K,)
            The amount of token-Y received across

        """

        for k in range(len(self.L)):
            # print(l[k], self.l[k])
            assert l[k] <= self.l[k], "you have insufficient LP tokens"

        # Compute the amount of token-X and token-Y you receive from each pool
        x = l*self.Rx/self.L
        y = l*self.Ry/self.L

        # Update the pool states
        self.Rx -= x
        self.Ry -= y
        self.L -= l
        self.l -= l
        
        return x, y
    
    def save_random_numbers(self, kappa, T, batch_size=256):
        N_list, event_type_list, event_direction_list, v_list = [], [], [], []
        # used for generating Poisson random variables for all events
        sum_kappa = np.sum(kappa)
        # used for thinning the Poisson process
        pi = kappa/sum_kappa

        for k in range(batch_size):
            N = np.random.poisson(lam = sum_kappa*T)
            N_list.append(N)
            
            for j in range(N):
                event_type0 = np.random.choice(len(kappa), p=pi)
                event_type_list.append(event_type0)

                event_direction0 = np.random.rand()
                event_direction_list.append(event_direction0)

                if event_type0 == 0:
                    # there is a swap across all venues
                    random_number0 = np.random.randn()
                    v_list.append(random_number0)
                else:
                    random_number0 = np.random.randn()
                    v_list.append(random_number0)

        N_list, event_type_list, event_direction_list, v_list = np.array(N_list), np.array(event_type_list), np.array(event_direction_list), np.array(v_list)

        # store all the arrays in a dictionary
        arrays_dict = {'N_list': N_list, 'event_type_list': event_type_list, 'event_direction_list': event_direction_list, 'v_random_number_list': v_list}
        # save the dictionary
        # np.save('random_numbers.npy', arrays_dict)

        return arrays_dict
    
    def simulate(self, kappa, p, sigma, T=1, batch_size=256):
        """
        Simulate trajectories of all AMM pools simultanesouly.

        Parameters
        ----------
        kappa : array (K+1,)
            rate of arrival of swap events X->Y and Y->X.
            kappa[0,:] is for a common event across all pools
        p : array (K+1,2)
            probability of swap X to Y event conditional on an event arriving.
            p[0,:] is for a common event across all pools
        sigma : array (K+1,2)
            standard deviation of log volume of swap events.
            sigma[0,:] is for a common event across all pools
        T : float, optional: default is 1.
            The amount of (calendar) time to simulate over
        batch_size : int, optional, default is 256.
            the number of paths to generate.

        Returns
        -------
        pools : deque, len=batch_size
            Each element of the list is the pool state at the end of the simulation for that scenario
        Rx_t : deque, len= batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the reserves in token-X for all AMM pools after each transaction.
        Ry_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the reserves in token-Y for all AMM pools after each transaction.
        v_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the volumes of the transaction sent to the various AMM pools -- the transaction is
            either a swap X for Y or swap Y for X for a single pool, or across all pools at once
        event_type_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the event type. event_type=0 if it is a swap sent to all pools simultaneously,
            otherwise, the swap was sent to pool event_type
        event_direction_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the directi of the swap.
            event_direction=0 if swap X -> Y
            event_direction=1 if swap Y -> X

        """

        # used for generating Poisson random variables for all events
        sum_kappa = np.sum(kappa)

        # used for thinning the Poisson process
        pi = kappa/sum_kappa

        # store the list of reservese generated by the simulation
        def make_list(batch_size):
          x = deque(maxlen=batch_size)
          x = [None] * batch_size
          return x

        Rx_t = make_list(batch_size)
        Ry_t = make_list(batch_size)       
        v_t = make_list(batch_size)
        event_type_t = make_list(batch_size)
        event_direction_t = make_list(batch_size)
        pools = make_list(batch_size)

        for k in tqdm(range(batch_size)):

            N = np.random.poisson(lam = sum_kappa*T)

            Rx = np.zeros((N,len(self.Rx)))
            Ry = np.zeros((N,len(self.Rx)))
            v = np.zeros((N,len(self.Rx)))
            event_type = np.zeros(N, int)
            event_direction = np.zeros(N, int)

            pools[k] = copy.deepcopy(self)

            for j in range(N):

                # generate the type of event associated with each event
                event_type[j] = np.random.choice(len(kappa), p=pi)

                # the direction of the swap 0 = x-> y, 1 = y -> x
                event_direction[j] = int(np.random.rand()< p[event_type[j]])

                if event_direction[j] == 0:
                    mu = np.zeros(len(pools[k].Rx)) # deposit X and get Y
                else:
                    mu = np.log(pools[k].Ry/pools[k].Rx) # deposit Y and get X

                if event_type[j] == 0:
                    # there is a swap across all venues
                    v[j,:] = np.exp((mu-0.5*sigma[0]**2) + sigma[0]*np.random.randn() )

                else:

                    # there is a swap only on a specific venue
                    v[j,:] = np.zeros(len(pools[k].Rx))
                    mu = mu[event_type[j]-1]
                    v[j,event_type[j]-1] = np.exp((mu-0.5*sigma[event_type[j]]**2) \
                                                   + sigma[event_type[j]]*np.random.randn() )

                if event_direction[j] == 0:
                    pools[k].swap_x_to_y(v[j,:]) # submit X and get Y
                else:
                    pools[k].swap_y_to_x(v[j,:]) # submit Y and get X

                Rx[j,:] = 1*pools[k].Rx
                Ry[j,:] = 1*pools[k].Ry

            Rx_t[k] = Rx
            Ry_t[k] = Ry
            v_t[k] = v
            event_type_t[k] = event_type
            event_direction_t[k] = event_direction

        return pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t
    
    def simulate_fast(self, kappa, p, sigma, T, N_list, event_type_list, event_direction_list, v_list,  batch_size=256):
        """
        This is an modified version of the simulate module that relies on previously generated random numbers
        via the save_random_numbers module. This has been done due to the problems arising with the use of
        cython. Specifically, it is hard (if not impossible) to synchronize the numpy random number generator
        between python and cython.

        Parameters
        ----------
        kappa : array (K+1,)
            rate of arrival of swap events X->Y and Y->X.
            kappa[0,:] is for a common event across all pools
        p : array (K+1,2)
            probability of swap X to Y event conditional on an event arriving.
            p[0,:] is for a common event across all pools
        sigma : array (K+1,2)
            standard deviation of log volume of swap events.
            sigma[0,:] is for a common event across all pools
        N_list : array (batch_size,)
            This is a list containing all the poisson random numbers.
        event_type_list : array 
            This is a list containing all the event types. The number of elements
            depends on the random numbers stored in N_list.
        event_direction_list : array
            This is a list containing all the random numbers used for the computation
            of event directions. The number of elements depends on the random numbers stored in N_list.
        v_list : array
            This is a list containing all the random numbers used for the computation
            of the swap volumes. The number of elements depends on the random numbers stored in N_list.
        T : float, optional: default is 1.
            The amount of (calendar) time to simulate over
        batch_size : int, optional, default is 256.
            the number of paths to generate.

        Returns
        -------
        pools : deque, len=batch_size
            Each element of the list is the pool state at the end of the simulation for that scenario
        Rx_t : deque, len= batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the reserves in token-X for all AMM pools after each transaction.
        Ry_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the reserves in token-Y for all AMM pools after each transaction.
        v_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the volumes of the transaction sent to the various AMM pools -- the transaction is
            either a swap X for Y or swap Y for X for a single pool, or across all pools at once
        event_type_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the event type. event_type=0 if it is a swap sent to all pools simultaneously,
            otherwise, the swap was sent to pool event_type
        event_direction_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the directi of the swap.
            event_direction=0 if swap X -> Y
            event_direction=1 if swap Y -> X

        """
        # used for generating Poisson random variables for all events
        sum_kappa = np.sum(kappa)
        # used for thinning the Poisson process
        pi = kappa/sum_kappa

        # store the list of reservese generated by the simulation
        def make_list(batch_size):
        #   x = deque(maxlen=batch_size)
          x = [None] * batch_size
          return x

        Rx_t = make_list(batch_size)
        Ry_t = make_list(batch_size)       
        v_t = make_list(batch_size)
        event_type_t = make_list(batch_size)
        event_direction_t = make_list(batch_size)
        pools = make_list(batch_size)

        
        for k in range(batch_size):
            N = N_list[k]
            
            Rx = np.zeros((N,len(self.Rx)))
            Ry = np.zeros((N,len(self.Rx)))
            v = np.zeros((N,len(self.Rx)))
            event_type = np.zeros(N, int)
            event_direction = np.zeros(N, int)

            pools[k] = copy.deepcopy(self)
            
            for j in range(N):
                if k == 0:
                    idx = 0
                else:
                    idx = N_list[:k].sum()
                event_type[j] = event_type_list[idx + j]
                event_direction[j] = int(event_direction_list[idx + j] < p[event_type[j]])

                if event_direction[j] == 0:
                    mu = np.zeros(len(pools[k].Rx)) # deposit X and get Y
                else:
                    mu = np.log(pools[k].Ry/pools[k].Rx) # deposit Y and get X
                if event_type[j] == 0:
                    # there is a swap across all venues
                    random_number = v_list[idx + j]
                    v[j,:] = np.exp((mu-0.5*sigma[0]**2) + sigma[0]*random_number)
                else:
                    # there is a swap only on a specific venue
                    v[j,:] = np.zeros(len(pools[k].Rx))
                    mu = mu[event_type[j]-1]
                    random_number = v_list[idx + j]
                    v[j,event_type[j]-1] = np.exp((mu-0.5*sigma[event_type[j]]**2) \
                                                   + sigma[event_type[j]]*random_number)
                if event_direction[j] == 0:
                    pools[k].swap_x_to_y(v[j,:]) # submit X and get Y
                else:
                    pools[k].swap_y_to_x(v[j,:]) # submit Y and get X
                Rx[j,:] = 1*pools[k].Rx
                Ry[j,:] = 1*pools[k].Ry

            Rx_t[k] = Rx
            Ry_t[k] = Ry
            v_t[k] = v
            event_type_t[k] = event_type
            event_direction_t[k] = event_direction

        return pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t

#%% Example of how to use the class

if __name__ == "__main__":
    Rx0 = np.array([100, 100, 100], float)
    Ry0 = np.array([1000, 1000, 1000], float)
    phi = np.array([0.003, 0.003, 0.003], float)

    pools = amm(Rx=Rx0, Ry=Ry0, phi=phi)
    print ('Available LP coins :', np.round(pools.L, 3))
    print('\n')

    y = pools.swap_x_to_y([1, 0.5, 0.1], quote=False)
    print('Obtained Y coins :', np.round(y, 2))
    print('Reserves in X :', np.round(pools.Rx, 2))
    print('Reserves in Y :', np.round(pools.Ry, 2))
    print('\n')

    x = [1, 1, 1]
    #l = pools.mint(x=x, y=np.random.rand(3))
    y = x*pools.Ry / pools.Rx
    l = pools.mint(x=x, y=y)
    print('Traded LP coins :', np.round(pools.l, 2))
    print('Pool LP coins :', np.round(pools.L, 2))
    print('\n')

    #x, y = pools.burn(l+1)
    x, y = pools.burn(l)
    print('Trader LP coins :', np.round(pools.l, 2))
    print('Trader X coins :', np.round(x, 2))
    print('Trader Y coins :', np.round(y, 2))
    print('Pool LP coins :', np.round(pools.L, 2))
    print('\n')

    l = pools.swap_and_mint([10, 10, 10])
    print('Minted LP coins :', np.round(l, 2))
    print('Total trader LP coins :', np.round(pools.l, 2))
    print('Available LP coins :', np.round(pools.L, 2))
    print('\n')

    total_x = pools.burn_and_swap(l)
    print('Number of X coins received :', np.round(total_x, 2))
    print('Total trader liquidity coins :', np.round(pools.l, 2))
    print('Available liquidity coins :', np.round(pools.L, 2))

# %%
