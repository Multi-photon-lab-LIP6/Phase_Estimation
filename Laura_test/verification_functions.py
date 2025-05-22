import numpy as np
from scipy.optimize import minimize

"""
The functions used in this script are based on the methods proposed in a verification protocol:
"Robust and efficient verification of graphy states in blind measurement-based quantum computation"

There you can find the equations refered to below
"""

### r_func is an inversion of eq. (19) such that we can have a fucntion that defines robustness
def r_func(epsilon, lamb, failure_rate, N):
    k = int(failure_rate * N)
    nu = 1 - lamb
    return (k / (N * nu * epsilon)) * (lamb * np.sqrt(2 * nu) + 1) - lamb * np.sqrt(2 * nu)

### r_bound returns 1-r. This will be used as bound on the robustness for the minimization method
### (r < 1) -> (1-r > 0)
def r_bound(epsilon, lamb, failure_rate, N):
    return 1 - r_func(epsilon, lamb, failure_rate, N)

### Returns the scalar epsilon that we want to minimize such that eq. (20) is true
def epsilon_func(epsilon):
    return epsilon

### Returns N_min that is defined in ineq (20), where the robustness is replaced by r_func
### This allows us to minimize epsilon without having to worry about r
def N_bound(epsilon, lamb, delta, failure_rate, N):
    nu = 1 - lamb
    r = r_func(epsilon, lamb, failure_rate, N)
    N_min = ((np.log(1/delta) + 4*lamb*nu**2)/epsilon)*((lamb*np.sqrt(2*nu) + r)/(lamb*nu*(1 - r)))**2
    return N_min

### We define the constraint on the minimization: N_experimental - N_min
### We will impose that this value should be >= 0
def const(epsilon, lamb, delta, failure_rate, N):
    return N - N_bound(epsilon, lamb, delta, failure_rate, N)

### Returns the result of a minimization of epsilon given:
### theoretical requirements: lamb, delta
### experimental params: N, failure_rate
def get_epsilon(lamb, delta, N, failure_rate, epsilon_ini_guess):
    func = epsilon_func #function to minimize
    # constranints are: 1. Eq (20) is verified; 2. r >= 0; 3. r <= 1
    cons = [{'type': 'ineq', 'fun': const, 'args': (lamb, delta, failure_rate, N)},
            {'type': 'ineq', 'fun': r_func, 'args': (lamb, failure_rate, N)},
            {'type': 'ineq', 'fun': r_bound, 'args': (lamb, failure_rate, N)}]

    epsilon_solution = minimize(func,
                                epsilon_ini_guess,
                                method='SLSQP',
                                bounds = [(0.,1.)],
                                constraints = cons)
    return epsilon_solution



class AnuGraph:

    def __init__(self, n_qubits, n_stabilizers):
        self.n_qubits = n_qubits
        self.n_stab = n_stabilizers

    def get_fidelity_bound(self, N_total, failure_rate, confidence_level):
        self.failure_rate = failure_rate
        self.confidence_level = confidence_level

        m = N_total/(2*self.n_stab**5*np.log(self.n_stab))
        c = 3*(1-np.log(1-self.confidence_level)/np.log(self.n_stab))/(2*m)

        if (c > 3/(2*m) and c < (self.n_stab-1)**2/4):
            self.fidelity_bound = 1 - 2*np.sqrt(c)/(self.n_stab)-2*self.n_stab*failure_rate
        else:
            self.fidelity_bound = 0

        return self.fidelity_bound
    
    def get_fidelity_bound_evolution(self, N_total_evol, failure_rate_evol, confidence_level):
        self.fidelity_bound_evol = []

        for (i, j) in zip(failure_rate_evol, N_total_evol):
            self.get_fidelity_bound(i, confidence_level, j)
        self.fidelity_bound_evol = np.array(self.fidelity_bound_evol, dtype = float)

        return self.fidelity_bound_evol
