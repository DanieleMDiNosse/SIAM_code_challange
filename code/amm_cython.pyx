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
        # Make sure the sizes of the input arrays are the same
        assert Rx.size == Ry.size and Ry.size == phi.size, "length of Rx, Ry, and phi must be the same."
        
        # Assign the input arrays to the class attributes
        self.Rx = 1*Rx
        self.Ry = 1*Ry
        self.phi = 1*phi
        self.N = len(self.Rx)
        
        self.L = np.sqrt(self.Rx * self.Ry)
        self.l = np.zeros(len(self.L), dtype=np.float64)

    def swap_x_to_y(self, cnp.ndarray x, bint quote=False):
        cdef cnp.ndarray y 
        y = x * self.Ry * (1 - self.phi) / (self.Rx + x * (1 - self.phi))
        if quote is False:
            self.Rx += x
            self.Ry -= y
        return y

    def swap_y_to_x(self, cnp.ndarray y, bint quote=False):
        x = y * self.Rx * (1 - self.phi) / (self.Ry + y * (1 - self.phi))
        if quote is False:
            self.Rx -= x
            self.Ry += y
        return x

    def mint(self, cnp.ndarray x, cnp.ndarray y):
        cdef int k, i
        cdef cnp.ndarray l 

        for k in range(self.Rx.shape[0]):
            assert fabs(((x[k]/y[k]) - self.Rx[k]/self.Ry[k])) < 1e-9, "pool " + str(k) + ...
        
        l = x*self.L/self.Rx
        for i in range(len(l)):
            if l[i] < 0:
                print(x[i], self.L[i], self.Rx[i])

        self.Rx += x
        self.Ry += y
        self.L += l
        self.l += l

        return l
    
    def swap_and_mint(self, cnp.ndarray x, bint quote=False):
        cdef cnp.ndarray theta
        cdef int k

        x = np.array(x, dtype=np.float64)
        theta = 1 + (2 - self.phi)*self.Rx*(1 - np.sqrt(
            1+4*x*(1-self.phi)/(self.Rx*(2-self.phi)**2)
            )) / ( 2*(1 - self.phi)*x )

        return self.mint(x * theta, self.swap_x_to_y(x * (1 - theta), quote=False))
    
    def burn_and_swap(self, cnp.ndarray l):
        cdef cnp.ndarray x, y, quote, to_swap
        cdef double total_x
        cdef int max_index

        x, y = self.burn(l)
        quote = self.swap_y_to_x(np.array([y.sum()] * self.N, dtype=np.float64), quote=True)
        to_swap = np.zeros_like(quote)

        max_index = np.argmax(quote)
        to_swap[max_index] = y.sum()

        total_x = (x + self.swap_y_to_x(to_swap, quote=False)).sum()
        return total_x

    def burn(self, cnp.ndarray l):
        cdef int k
        cdef cnp.ndarray x, y

        for k in range(self.L.shape[0]):
            assert l[k] <= self.l[k], "you have insufficient LP tokens"

        x = l * self.Rx / self.L
        y = l * self.Ry / self.L

        self.Rx -= x
        self.Ry -= y
        self.L -= l
        self.l -= l

        return x, y

    
    def simulate(self, cnp.ndarray kappa, cnp.ndarray p, cnp.ndarray sigma, double T, cnp.ndarray N_list, cnp.ndarray event_type_list, cnp.ndarray event_direction_list, cnp.ndarray v_list, int batch_size=256):
        cdef double sum_kappa
        cdef cnp.ndarray pi, mu_vec, event_type, event_direction
        cdef double mu
        cdef int k, j, N, c, random_number
        cdef cnp.ndarray Rx, Ry, v
        cdef list pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t
        sum_kappa = np.sum(kappa)
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