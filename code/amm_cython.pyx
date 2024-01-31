# cython: language_level=3
cimport numpy as cnp
from libc.math cimport fabs, sqrt, exp, log
import numpy as np
from params import params
from libc.stdlib cimport rand, RAND_MAX
from collections import deque
import copy

cdef class amm_cython:
    cdef public:
        cnp.ndarray Rx, Ry, L, l, phi
        int N
    
    def __init__(self, cnp.ndarray Rx, cnp.ndarray Ry, cnp.ndarray phi):
        """
        Instantiate the class

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
        # Make sure the sizes of the input arrays are the same
        assert Rx.size == Ry.size and Ry.size == phi.size, "length of Rx, Ry, and phi must be the same."
        
        # Assign the input arrays to the class attributes
        self.Rx = 1*Rx
        self.Ry = 1*Ry
        self.phi = 1*phi
        self.N = len(self.Rx)
        
        # number of LP tokens for each pool
        self.L = np.sqrt(self.Rx * self.Ry)
        # the trader begins with no LP tokens
        self.l = np.zeros(len(self.L), dtype=np.float64)

    def swap_x_to_y(self, cnp.ndarray x, bint quote=False):
        """

        Swap token-X for token-Y across all pools simulataneously

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
        cdef cnp.ndarray y 
        # Compute the amount of token-Y you receive from each pool
        y = x * self.Ry * (1 - self.phi) / (self.Rx + x * (1 - self.phi))

        # If quote is False, then update the pool states
        if quote is False:
            self.Rx += x
            self.Ry -= y
        return y

    def swap_y_to_x(self, cnp.ndarray y, bint quote=False):
        """
        Swap token-Y for token-X across all pools simulataneously

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
        x = y * self.Rx * (1 - self.phi) / (self.Ry + y * (1 - self.phi))

        # If quote is False, then update the pool states
        if quote is False:
            self.Rx -= x
            self.Ry += y
        return x

    def mint(self, cnp.ndarray x, cnp.ndarray y):
        """

        Mint LP tokens across all pools

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
        cdef int k, i
        cdef cnp.ndarray l 

        for k in range(self.Rx.shape[0]):
            assert fabs(((x[k]/y[k]) - self.Rx[k]/self.Ry[k])) < 1e-9, "pool " + str(k) + ...
        
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
    
    def swap_and_mint(self, cnp.ndarray x, bint quote=False):
        """
        This is a method that determines the correct amount of y for each x within the corresponding pool
        to swap and then mint tokens with the reamaing x and the y you receive.

        Parameters
        ----------
        x : array (K,)
            amount of token-X you have for each pool.

        Returns
        -------
        l : array (K,)
            The amount of LP tokens you receive from each pool.
        """
        cdef cnp.ndarray theta
        cdef int k

        # Compute the percentage to swap in each pool
        x = np.array(x, dtype=np.float64)
        theta = 1 + (2 - self.phi)*self.Rx*(1 - np.sqrt(
            1+4*x*(1-self.phi)/(self.Rx*(2-self.phi)**2)
            )) / ( 2*(1 - self.phi)*x )

        return self.mint(x * theta, self.swap_x_to_y(x * (1 - theta), quote=False))
    
    def burn_and_swap(self, cnp.ndarray l):
        """
        A method that burns your LP tokens, then swaps y to x and returns only x

        Parameters
        ----------
        l : array (K,)
            amount of LP tokens you have for each pool.

        Returns
        -------
        x : array (K,)
            The amount of token-x you receive at the end.
        """
        cdef cnp.ndarray x, y, quote, to_swap
        cdef double total_x
        cdef int max_index

        x, y = self.burn(l)
        # Request quote
        quote = self.swap_y_to_x(np.array([y.sum()] * self.N, dtype=np.float64), quote=True)
        # Assign y to the pool with the highest quote
        to_swap = np.zeros_like(quote)

        max_index = np.argmax(quote)
        to_swap[max_index] = y.sum()

        # Swap coin
        total_x = (x + self.swap_y_to_x(to_swap, quote=False)).sum()
        return total_x

    def burn(self, cnp.ndarray l):
        """
        Burn LP tokens across all pools

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
        cdef int k
        cdef cnp.ndarray x, y

        for k in range(self.L.shape[0]):
            assert l[k] <= self.l[k], "you have insufficient LP tokens"

        # Compute the amount of token-X and token-Y you receive from each pool
        x = l * self.Rx / self.L
        y = l * self.Ry / self.L

        # Update the pool states
        self.Rx -= x
        self.Ry -= y
        self.L -= l
        self.l -= l

        return x, y

    
    def simulate(self, cnp.ndarray kappa, cnp.ndarray p, cnp.ndarray sigma, double T, cnp.ndarray N_list, cnp.ndarray event_type_list, cnp.ndarray event_direction_list, cnp.ndarray v_list, int batch_size):
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
        batch_size : int
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
        cdef double sum_kappa
        cdef cnp.ndarray pi, mu_vec, event_type, event_direction
        cdef double mu
        cdef int k, j, N, c, random_number
        cdef cnp.ndarray Rx, Ry, v
        cdef list pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t
        # used for generating Poisson random variables for all events
        sum_kappa = np.sum(kappa)
        # used for thinning the Poisson process
        pi = kappa / sum_kappa

        pools = [None] * batch_size

        for k in range(batch_size):
            N = N_list[k]

            v = np.zeros((N, len(self.Rx)))
            event_type = np.zeros(N, dtype=int)
            event_direction = np.zeros(N, dtype=int)

            pools[k] = copy.deepcopy(self)

            for j in range(N):
                if k == 0:
                    idx = 0
                else:
                    # This is the correct index to choose based on the dynamics of save_random_numbers in amm.py
                    idx = N_list[:k].sum()

                event_type[j] = event_type_list[idx + j]
                
                event_direction[j] = int(event_direction_list[idx + j] < p[event_type[j]])
                
                if event_direction[j] == 0:
                    mu_vec = np.zeros(len(pools[k].Rx))
                else:
                    mu_vec = np.log(pools[k].Ry / pools[k].Rx)
                if event_type[j] == 0:
                    v[j, :] = np.exp((mu_vec - 0.5 * sigma[0]**2) + sigma[0] * v_list[idx + j])
                else:
                    v[j, :] = np.zeros(len(pools[k].Rx))
                    mu = mu_vec[event_type[j] - 1]
                    v[j, event_type[j] - 1] = np.exp((mu - 0.5 * sigma[event_type[j]]**2) + sigma[event_type[j]] * v_list[idx + j])
                if event_direction[j] == 0:
                    pools[k].swap_x_to_y(v[j, :])
                else:
                    pools[k].swap_y_to_x(v[j, :])

        return pools
