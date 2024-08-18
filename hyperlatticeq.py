import math
import time
import copy
import numpy as np
import networkx as nx
from itertools import combinations, product

import warnings
warnings.filterwarnings('ignore')
np.seterr(invalid='ignore')


class General_HyperlatticeQ(object):
    """
    Hyperlattice Queuing Model with Infinite-line Capacity

    A generalized hypercube queuing model, called Hyperlattice, that captures 
    the dynamics of emergency response operations with overlapping patrol. The 
    system state depends on the status of all the servers in this system, and 
    the number of calls in each queue to be processed. These states can be 
    represented by a hyperlattice in dimension I. Each node of the hyperlattice 
    corresponds to a state B = (ni)_{i in set_I} represented by a tuple of 
    numbers, where non-negative integer ni indicates the status of server i. 
    Server i is idle if ni = 0 and busy if ni > 0. The value of ni - 1 
    represents the number of calls waiting in the queue of server i when the 
    server is busy. 
    
    We note that the queuing system in the aggregate is simply a birth-death 
    process whose states are represented by Nk, where k=0,1,...,infinity. This 
    equivalence is obtained by collecting together all states having equal value
    of k = \norm{B}_1. Their summed probability of occurrence is equal to the 
    comparable probability of state occurring in the birth-death model. 
    """

    def __init__(self, I, Lam, Eta, mu, K=5, inf=30):
        """
        Params:
        * I:    number of servers (service regions)
        * Lam:  arrival rates of each sub-regions, which has the same shape as 
                the adjacent matrix A. The values on the diagonal (i, i)  
                indicate the arrival rates of primary service regions and values 
                on the off-diagonal (i, j) and i != j indicates the arrival 
                rates of overlapping service regions. 
        * Eta:  dispatch probability when both servers are available or busy, 
                which has the same shape as the adjacent matrix A. The value of 
                entry (i, j) represents the probability of assigning call to 
                server i in the overlapping region (i, j). 
                Note that Eta_ij + Eta_ji = 1. 
        * mu:   service rates of each server
        * K:    number of aggregated states to be truncated (estimated)
        * inf:  the maximum length of the queue considered in the model.
                Note that inf >> K is required. 
        """
        # hyperparameters
        self.I = I
        self.K = K                                                   
        self.n_est_states = sum([ math.comb(I+k-1,I-1) for k in range(K+1) ])   # number of states in the truncated hyperlattice (need to be estimated through solving balance equations)
        self.set_B        = self._tour(K)                                       # set of states in the truncated hyperlattice
        # model parameters
        self.mu     = mu                                                        # service rates
        self.Lam    = Lam                                                       # arrival rates as a dict with respect to h_k
        self.Eta    = Eta                                                       # dispatch policy
        time_0 = time.time()
        self.Q      = self._transition_rates()                                  # transition rate matrix
        time_1 = time.time()
        self.sumLam = sum(self.Lam.values())                                    # sum of arrival rates of the entire service system
        time_2 = time.time()
        self.aggLam = np.array([math.comb(k+I-1, k) * self.sumLam for k in range(inf + 1)])  
        time_3 = time.time()
        self.aggLam_formula = np.array([math.comb(k+I-1, k) * self.sumLam for k in range(inf + 1)])  
        time_4 = time.time()
        self.aggMu  = np.array([ math.comb(k+I-1, k) * I * mu
                                for k in range(inf + 1) ])                      # aggregated service rates (bar_mu_k)  index mathematically start from 1
        time_5 = time.time()
        # steady-state distribution
        self.Pi_N   = self._birth_death_solver(inf)                             # steady state probabilities of aggregated states (Nk)
        time_6 = time.time()
        self.Pi_B   = self._balance_equations_solver()                          # steady state probabilities of states in the truncated hyperlattice (Bu)
        time_7 = time.time()
        self.Pi_B_power = self._balance_equations_solver_power_method()
        time_8 = time.time()
    

    def _tour(self, K):
        """
        Tour Algorithm

        The construction of the transition rate matrix requires a complete tour 
        of all the states in the hyperlattice. The tour algorithm generates a 
        complete sequence B_0, B_1, ... of I-digit non-negative integer numbers, 
        with infinite members in the sequence.
        """
        def binary2states(I, bvec):                                             # translate binary combinatorics into state (a vector of non-negative integers)
            B = np.zeros(I)                                                     # initialization of state
            i = 0                                                               # server index
            for b in bvec:
                if b == 1:
                    B[i] += 1                                                   # n_i + 1 if the "ball" is selected
                else:
                    i += 1                                                      # otherwise move to the next ball
            return B

        # initialization
        u     = 0                                                               # state index
        nB    = sum([ math.comb(self.I+k-1,self.I-1) for k in range(K+1) ])     # number of states in the tour
        set_B = np.zeros((nB, self.I))                                          # set of states in the truncated hyperlattice
        # tour starting from N0 to NK
        for k in range(K+1):
            combs = combinations(np.arange(k+self.I-1), k)                      # separate k stars (using I-1 bars) into I groups
            for comb in combs:
                bvec             = np.zeros(k+self.I-1)                         # combinatorics
                bvec[list(comb)] = 1
                set_B[u]         = binary2states(self.I, bvec)                  # state
                u               += 1
        return set_B                                                            # set of all states with shape [ n_est_states, I ]
    
        
    def _prob(self, Bu, cur_set, i):
        """
        Probability mass function p_hk.
        
        This function gives the probability of the new call in the primary 
        overlapping service region "cur_set" being served by server i
        
        The current policy is to divide the probability according to the eta matrix
        """
        prob_dist = self.Eta[cur_set]
        cur_set_len = len(cur_set)
        
        if sum(prob_dist) == 0: # if the current set overlapping does not exist
            return 0

        busy_index = []
        idle_index = []
        for j in range(cur_set_len):
            if Bu[cur_set[j]] > 0:
                busy_index.append(cur_set[j])
            else:
                idle_index.append(cur_set[j])

        if Bu[i] > 0 and len(idle_index) > 0: # case 1) i is busy and not all other servers are busy
            return 0
        elif Bu[i] == 0 and len(idle_index) == 1: # case 2) i is idle and all other servers are busy
            return 1
        elif Bu[i] == 0 and cur_set_len > len(idle_index) > 1: # case 3) i is idle and some other servers are idle
            allocate_ratio = sum([prob_dist[k] for k in busy_index])
            remain_ratio = sum([prob_dist[k] for k in idle_index])
            new_prob_dist = np.array(prob_dist)
            new_prob_dist[busy_index] = 0
            for j in idle_index:
                new_prob_dist[j] += allocate_ratio * (new_prob_dist[j] / remain_ratio)
            assert sum(new_prob_dist) - 1 < 1e-5 # test if new_prob_dist is a probability distribution
            return new_prob_dist[i]
        else: # case 4) all servers are busy or idle
            return prob_dist[i]

    def __q(self, Bu, Bv):
        """
        Return the Transition Rate Between State Bu and Bv
        """
        quv  = 0
        dvec = Bu - Bv
        if np.linalg.norm(dvec, ord=1) == 1 and sum(dvec) > 0:                  # downward transition rate
            quv     = self.mu
        elif np.linalg.norm(dvec, ord=1) == 1 and sum(dvec) < 0:                # upward transition rate
            i       = np.where(dvec == -1)[0][0]                                # index of the state becoming busy
            for cur_set in self.Eta.keys():
                if i in cur_set:
                    quv += self._prob(Bu, cur_set, i)*self.Lam[cur_set]
        else:
            quv = 0.
        return quv
    
    def _transition_rates(self):
        """
        Generate Transition Rate Matrix
        """
        # initialization of transition rate matrix
        Q = np.zeros((self.n_est_states, self.n_est_states))                    
        # fill in off-diagonal entries
        for u in range(self.n_est_states):
            for v in range(self.n_est_states):
                Q[u, v] = self.__q(self.set_B[u], self.set_B[v])                # transition rate between state Bu and Bv
        # fill in diagonal entries
        for u in range(self.n_est_states):
            Q[u, u] = - Q[u, :].sum()                                           # so that each row sums to 0
        return Q                                                                # transition rate matrix with shape [ n_est_states, n_est_states ]

    def _aggregated_arrival_rates(self, inf):
        """
        Arrival Rates for Aggregated States Nk

        TODO:
        Optimize the calculation of aggregated arrival rates. 
        """
        # find state indices for each aggregated group
        ids     = []                                                            
        init_id = 0
        for k in range(inf + 1):
            idk     = np.arange(math.comb(self.I+k-1, self.I-1)) + init_id      # ids[k] gives state indices within the aggregated state k
            init_id = idk[-1] + 1
            ids.append(idk)
        # calculate sum of arrival rates between two consecutive groups
        set_B  = self._tour(inf)                                                # set of all the states 
        aggLam = []
        for k in range(inf):
            agglam = 0
            for uk, uk1 in product(ids[k], ids[k+1]):                           # for a state (uk) from Nk and another state (uk1) from Nk+1
                qukuk1 = self.__q(set_B[uk], set_B[uk1])                        # this step is computationally intensive
                if qukuk1 > 0:
                    agglam += qukuk1
            aggLam.append(agglam)
        return np.array(aggLam)
            
    def aggregated_arrival_rates_saturated(self, inf):
        """
        Arrival Rates for Aggregated States Nk (Only Between Saturated States)

        This function calculates the sum of all upward transition rates between
        saturated states in Nk and saturated states in Nk+1.

        TODO:
        Optimize the calculation of aggregated arrival rates. 
        """
        set_B   = self._tour(inf)                                               # set of all the states 
        # find state indices for each aggregated group
        ids     = []                                                            
        init_id = 0
        for k in range(inf + 1):
            idk     = np.arange(math.comb(self.I+k-1, self.I-1)) + init_id      # ids[k] gives state indices within the aggregated state k
            idk_sat = [ u for u in idk if sum(set_B[u] > 0) == self.I ]         # ids_sat[k] gives saturated state indices within the aggregated state k
            init_id = idk[-1] + 1
            ids.append(idk_sat)
        # calculate sum of arrival rates between two consecutive groups
        aggLam = []
        for k in range(inf):
            agglam = 0
            for uk, uk1 in product(ids[k], ids[k+1]):                           # for a state (uk) from Nk and another state (uk1) from Nk+1
                qukuk1 = self.__q(set_B[uk], set_B[uk1])                        # this step is computationally intensive
                if qukuk1 > 0:
                    agglam += qukuk1
            aggLam.append(agglam)
        return np.array(aggLam)
    
    def _birth_death_solver(self, inf):
        """
        Birth-Death Model Solver 

        This function calculates steady state probabilities for aggregated 
        states in a birth-death model. 
        """
        C    = np.array([                                                       # C0 = 1
            1 if k == 0 \
            else np.prod([ self.aggLam[n]/self.aggMu[n+1] for n in range(k) ])  # Ck = ( lam_0 * ... * lam_k-1 ) / ( mu_1 * ... * mu_k )
            for k in range(inf) ])
        Pi_N = np.array([                                                       # P0 = 1 / ( sum_k=1^inf Ck )
            1 / C.sum() if k == 0 else C[k] / C.sum()                           # Pk = C_k / ( sum_k=1^inf Ck )
            for k in range(inf) ])
        return Pi_N

    def _balance_equations_solver(self):
        """
        Solve Balance Equations by Matrix Inversion

        This function calculates steady state probabilities for states in the 
        truncated hyperlattice.

        Reference: 
        Constructing and Solving Markov Processes, Section 3.3.2
        https://homepages.inf.ed.ac.uk/jeh/biss2013/Note3.pdf

        TODO: 
        Implement power method to solve balance equations more efficiently, 
        especially when I is large. 
        """
        sum_Pi   = self.Pi_N[:self.K+1].sum()
        Q        = self.Q.copy()                                                # create a copy of transition rate matrix
        Q[:, -1] = np.ones(self.n_est_states)                                   # replace the last column with a vector of 1's
        en       = np.zeros(self.n_est_states)                                  # initialize “solution” vector, which was all zeros to be a column vector with 1 in the last row, and zeros everywhere else
        en[-1]   = sum_Pi
        invQ     = np.linalg.inv(Q.transpose())                                 # inverse of matrix Q
        Pi       = np.matmul(invQ, en)                                          # Pi = (Q^T)^{-1} * en
        return Pi
    
    def _balance_equations_solver_power_method(self, thres = 1e-3):
        '''
        Solve Balance Equations by Power Method
        Convergence threshold is given by input 'thres'
        '''
        gamma = -min(self.Q.diagonal())
        Pi = np.zeros(self.n_est_states)
        Pi[0] = 1
        next_Pi = np.matmul(Pi, np.identity(self.n_est_states)+self.Q/gamma)
        while max(np.absolute(next_Pi - Pi)) > thres:
            Pi = next_Pi
            next_Pi = np.matmul(Pi, np.identity(self.n_est_states)+self.Q/gamma)
        return next_Pi
    

class General_HLQperformance(object):
    """
    This class implements a number of key performance measures for a 
    hyperlattice queue model, including individual workloads, fraction of 
    dispatches, and average travel time, etc. 
    """

    def __init__(self, hq, T=None):
        """
        Params:
        * hq: hyperlattice queue model
        * T:  travel time matrix with shape
        """
        self.hq  = hq
        self.T   = T                                                            # travel time matrix (t_ij represents mean travel time of server i from region i to region (i,j))
        self.rho = self._individual_workloads()                                 # individual workloads for all servers
        self.Rho = self._fraction_of_dispatches()                               # fraction of dispatches for server i in region (i,j)
        if T is not None:
            self.t   = self._unconditional_mean_travel_time(self.T, self.Rho)       # unconditional mean travel time
            self.t_server = self._mean_server_travel_time(self.T, self.Rho)         # mean travel time of each server
            self.t_region = self._mean_region_travel_time(self.T, self.Rho)

    def _individual_workloads(self):
        """
        Return individual workloads for all servers in the system.
        """
        rho = np.zeros(self.hq.I) 
        for B, pi_B in zip(self.hq.set_B, self.hq.Pi_B):
            for i in range(self.hq.I):
                if B[i] > 0: 
                    rho[i] += pi_B
        return rho
    
    def _fraction_of_dispatches(self):
        """
        Return fraction of dispatches that send a server to each sub-region 
        under its responsibility. The sum of all fractions equals 1.
        """
        Rho = {}
        for cur_set in self.hq.Eta.keys():
            cur_set_frac = {}
            for i in cur_set:
                if len(cur_set) == 1:
                    cur_set_frac[i] = self.hq.Lam[cur_set] / self.hq.sumLam
                else:
                    rho = 0
                    for B, pi_B in zip(self.hq.set_B, self.hq.Pi_B):
                        rho += self.hq._prob(B, cur_set, i) * self.hq.Lam[cur_set] * pi_B 
                    cur_set_frac[i] = rho / self.hq.sumLam
            Rho[cur_set] = cur_set_frac
        return Rho
    
    def _unconditional_mean_travel_time(self, T, Rho):
        """
        Return unconditional mean travel time.
        """
        time = 0
        for cur_set in Rho.keys():
            for cur_server in Rho[cur_set].keys():
                time += Rho[cur_set][cur_server] * T[cur_set][cur_server]         
        return time
    
    def _mean_server_travel_time(self, T, Rho):
        """
        Return mean travel time of each server.
        """
        numerator = np.zeros(self.hq.I)
        denominator = np.zeros(self.hq.I)
        for cur_set in Rho.keys():
            for cur_server in Rho[cur_set].keys():
                denominator[cur_server] += Rho[cur_set][cur_server]
                numerator[cur_server] += Rho[cur_set][cur_server] * T[cur_set][cur_server]
        return numerator / denominator

    def _mean_region_travel_time(self, T, Rho):
        """
        Return mean travel time to overlapping region (i,j).
        """
        numerator = np.zeros(len(Rho))
        denominator = np.zeros(len(Rho))
        index = 0
        time = {}
        for cur_set in Rho.keys():     
            for cur_server in Rho[cur_set].keys():
                denominator[index] += Rho[cur_set][cur_server]
                numerator[index] += Rho[cur_set][cur_server] * T[cur_set][cur_server]  
            time[cur_set] = numerator[index] / denominator[index]
            index += 1
        return time
    

class allocation(object):
    """
    This class convert and evaluate an allocation solution under case 1 model, 
    where the regions and their arrival rate lambda are fixed.
    """
    
    def __init__(self, I, lambda_list, A, mu, K, inf):
        """
        Params:
        * I:            number of servers
        * lambda_list:  the list of arrival rate for each region and subregions
        * eta_matrix:   the matrix that each colunm are the servers and row are the regions/subregions,
                        each row stands for the probability distribution of a call in that region being
                        served by different servers.
        * mu:   service rates of each server
        * K:    number of aggregated states to be truncated (estimated)
        * inf:  the maximum length of the queue considered in the model.
                Note that inf >> K is required. 
        """
        self.I = I
        self.A = A
        self.N = len(lambda_list)
        self.lambda_list = lambda_list
        self.mu = mu
        self.K = K
        self.inf = inf
        self.T = None
        self.lat_dict = {}
        self.idx_dict = {}

    def T_mat_2_dict(self, T_matrix, regions):
        T = {}
        for cur_set in regions:
            time_dict = {}
            for j in cur_set: # for server j
                time = 0
                for i in cur_set: # for each sub-region i
                    time += (self.lambda_list[i]/sum([self.lambda_list[k] for k in cur_set])) * T_matrix[j][i]
                time_dict[j] = time
            T[cur_set] = time_dict
        return T
        
    def workload_std(self, eta_matrix):
        self.eta_matrix = eta_matrix
        self.lambda_dict = self._get_lambda_dict()
        self.eta_dict = self._get_eta_dict()
        self.check_feasibility()
        
        self.hq = General_HyperlatticeQ(I=self.I, Lam=self.lambda_dict, Eta=self.eta_dict, 
                                        mu=self.mu, K=self.K, inf=self.inf)
        self.perf = General_HLQperformance(self.hq)
        self.rho_std = np.std(self.perf.rho)
        return self.rho_std * 100 # percent%
    
    def mean_travel_time(self, eta_matrix, T):
        self.eta_matrix = eta_matrix
        self.lambda_dict = self._get_lambda_dict()
        self.eta_dict = self._get_eta_dict()
        self.check_feasibility()
        self.T = self.T_mat_2_dict(T, self.eta_dict.keys())
        
        self.hq = General_HyperlatticeQ(I=self.I, Lam=self.lambda_dict, Eta=self.eta_dict, 
                                        mu=self.mu, K=self.K, inf=self.inf)
        self.perf = General_HLQperformance(self.hq, self.T)
        mrt = self.perf.t
        mrt_server = self.perf.t_server
        mrt_region = self.perf.t_region
        return mrt, mrt_server, mrt_region
    

    def mean_travel_time_real_case(self, eta_matrix, T):
        self.eta_matrix = eta_matrix
        self.lambda_dict = self._get_lambda_dict()
        self.eta_dict = self._get_eta_dict()
        self.check_feasibility()
        self.T = self._get_T_dict(T)
        
        self.hq = General_HyperlatticeQ(I=self.I, Lam=self.lambda_dict, Eta=self.eta_dict, 
                                        mu=self.mu, K=self.K, inf=self.inf)
        self.perf = General_HLQperformance(self.hq, self.T)
        mrt = self.perf.t
        mrt_server = self.perf.t_server
        mrt_region = self.perf.t_region
        return mrt, mrt_server, mrt_region

    
    def check_feasibility(self):
        """
        Check feasibility for the current solution
        """
        for cur_region in self.eta_matrix:
            assert sum(cur_region) - 1 < 1e-5 or sum(cur_region) == 0
            
        for cur_server in self.eta_matrix.transpose():
            region_list = []
            for cur_region_index in range(self.N):
                if cur_server[cur_region_index] > 0:
                    region_list.append(cur_region_index)
                    
            adjacent = 1
            for i in range(len(region_list)):
                for j in range(i+1, len(region_list)):
                    adjacent = adjacent * self.A[region_list[i], region_list[j]]
            assert adjacent == 1
    
    def _get_lambda_dict(self):
        """
        Convert the lambda matrix to the input format of the hyperlattice
        """
        lambda_dict = {}
        for cur_region_index in range(self.N):
            cur_region = self.eta_matrix[cur_region_index]
            server_list = []
            for cur_server_index in range(self.I):
                if cur_region[cur_server_index] > 0:
                    server_list.append(cur_server_index)
            
            if tuple(server_list) not in lambda_dict:
                lambda_dict[tuple(server_list)] = self.lambda_list[cur_region_index]
            else:
                lambda_dict[tuple(server_list)] += self.lambda_list[cur_region_index]

            self.lat_dict[tuple(server_list)] = cur_region_index
            if len(server_list) == 1:
                self.idx_dict[server_list[0]] = cur_region_index

        return lambda_dict
    
    def _get_eta_dict(self):
        """
        Convert the eta matrix to the input format of the hyperlattice
        """
        eta_dict = {}
        cur_lambda_dict = {}
        for cur_region_index in range(self.N):
            cur_region = self.eta_matrix[cur_region_index]
            server_list = []
            for cur_server_index in range(self.I):
                if cur_region[cur_server_index] > 0:
                    server_list.append(cur_server_index)
                    
            if tuple(server_list) not in eta_dict:
                eta_dict[tuple(server_list)] = self.eta_matrix[cur_region_index]
                cur_lambda_dict[tuple(server_list)] = self.lambda_list[cur_region_index]
            else:
                ratio = self.lambda_list[cur_region_index]/(self.lambda_list[cur_region_index] + cur_lambda_dict[tuple(server_list)])
                eta_dict[tuple(server_list)] = (1-ratio)*eta_dict[tuple(server_list)] + ratio*self.eta_matrix[cur_region_index]
                assert sum(eta_dict[tuple(server_list)]) - 1 < 1e-5
        return eta_dict
    
    def _get_T_dict(self, T_matrix):
        """
        Convert the eta matrix to the input format of the hyperlattice
        """
        T_dict = {}
        cur_lambda_dict = {}
        for cur_region_index in range(self.N):
            cur_region = self.eta_matrix[cur_region_index]
            server_list = []
            for cur_server_index in range(self.I):
                if cur_region[cur_server_index] > 0:
                    server_list.append(cur_server_index)
                    
            if tuple(server_list) not in T_dict:
                cur_T_dict = {}
                for i in server_list:
                    cur_T_dict[i] = T_matrix[i][cur_region_index]
                T_dict[tuple(server_list)] = cur_T_dict
                cur_lambda_dict[tuple(server_list)] = self.lambda_list[cur_region_index]
            else:
                ratio = self.lambda_list[cur_region_index]/(self.lambda_list[cur_region_index] + cur_lambda_dict[tuple(server_list)])
                T_dict[tuple(server_list)] = {i: (1-ratio)*T_dict[tuple(server_list)][i] + ratio*T_matrix[cur_region_index][i] for i in server_list}
        return T_dict
    

### Simulation ###

def get_neighbour(old_eta_matrix, A, thres=0.25, is_print=False):
    '''
    Return a neighbour of a given eta matrix, which could be one of the following:
    1) collapse: reduce the number of servers of a random region, remove the surplus probability to another server
    2) reallocate: reallocate the probability distribution of a random region
    3) expand: increase the number of servers of a random region and reallocate the probability distribution
    '''
    eta_matrix = copy.deepcopy(old_eta_matrix)
    (N, I) = eta_matrix.shape
    collapse_index = np.sum(eta_matrix > 0, axis = 1) > 1
    collapse_index = np.where(collapse_index == True)[0]
    
    if len(collapse_index) > 0 and np.random.rand() < thres:
        # collapse
        if is_print:
            print('collapse')
        region_index = np.random.choice(collapse_index)
        region = eta_matrix[region_index]
        index = np.where(region>0)[0]
        collapse_server_index = np.random.choice(index, 2, replace=False)
        
        value = region[collapse_server_index[0]]
        region[collapse_server_index[1]] += value
        region[collapse_server_index[0]] = 0
        
    else:
        # expend or reallocate probability
        G = nx.from_numpy_array(A)
        cliques = np.array(list(nx.find_cliques(G)))
        
        # random pick a region
        region_index = np.random.randint(N)
        region = eta_matrix[region_index]
        busy_servers = np.where(region>0)[0]
        idle_servers = np.where(region==0)[0]
        
        if len(idle_servers) == 0:
            if is_print:
                print('reallocate')
            new_server_index = busy_servers
        else:
            np.random.shuffle(idle_servers)
            expend = False
            for cur_server in idle_servers:
                related_regions = eta_matrix.transpose()[cur_server]
                related_regions_index = np.append(np.where(related_regions>0)[0], region_index)
                feasible = any(all(i in b for i in related_regions_index) for b in cliques)
                if feasible:
                    new_server_index = np.append(busy_servers, cur_server)
                    expend = True
                    if is_print:
                        print('expand')
                    break
            if not expend:
                if is_print:
                    print('reallocate')
                new_server_index = busy_servers
                    
        partial_row = np.random.rand(len(new_server_index))
        partial_row = partial_row/partial_row.sum()
        region[new_server_index] = partial_row
        
    return eta_matrix

def local_neighbours(old_eta_matrix, A, K = 5):
    X = get_neighbour(old_eta_matrix, A)
    for k in range(K-1):
        X = get_neighbour(X, A)
    return X

class Eta_simulator(object):
    def __init__(self, N, I, A):
        self.N = N
        self.I = I
        self.A = A
        self.G = nx.from_numpy_array(A)
        self.cliques = np.array(list(nx.find_cliques(self.G)))
        
    def _simulate(self):
            '''
            Simulate an Eta matrix that follows the adjacent matrix A
            '''
            random_eta_matrix_T = np.zeros((self.I, self.N))
            counter = 0

            # for each server, pick a subset of a clique of regions that it serves
            # not randomly pick, pick in sequence so that all the regions are covered
            for cur_row in random_eta_matrix_T:
                if counter < len(self.cliques):
                    clique = self.cliques[counter]
                else:
                    clique_index = np.random.choice(len(self.cliques))
                    clique = self.cliques[clique_index]
                size = len(clique)
                index = np.random.choice(clique, size, replace=False)
                cur_row[index] = np.ones(size)
                counter += 1

            # for each region, random allocate the probability of being served by the servers
            random_eta_matrix = random_eta_matrix_T.transpose()
            for cur_row in random_eta_matrix:
                size = sum(cur_row == 1)
                index = (cur_row == 1)
                partial_row = np.random.rand(size)
                partial_row = partial_row/partial_row.sum()
                cur_row[index] = partial_row
            return random_eta_matrix
        
    def simulate(self, K=5):
        '''
        some perturbation
        '''
        eta = self._simulate()
        for i in range(K):
            eta = get_neighbour(eta, self.A)
        eta_T = eta.transpose()
        eta = np.random.permutation(eta_T).transpose()
        return eta
    
    def batch_simulate(self, n):
        simulations = np.array([self.simulate()])
        for i in range(n-1):
            simulations = np.append(simulations, np.array([self.simulate()]), axis=0)
        return simulations