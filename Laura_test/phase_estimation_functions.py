import numpy as np
import scipy.optimize as sp
import numdifftools as nd
import functools as ft
from scipy.linalg import block_diag

def get_analytical_hessian_sumfunction(n_qubits, x, v, f_x0, f_x1):
    '''
    Function returns the likelihood function of the probability distribution
    theta : theta maximum likelihood estimation
    v : visibility
    f_xi: is the frequency of the measurement result x=i 
    '''
    hessian_matrix = np.zeros((n_qubits, n_qubits))
    theta = np.sum(x)

    d2dt_x0 = (-v*(v+np.cos(theta))/((v*np.cos(theta)+1)**2))
    d2dt_x1 = (v*(-v+np.cos(theta))/((1-v*np.cos(theta))**2))

    for i in range(4):
        for j in range(4):
            hessian_matrix[i][j] = - (f_x0*d2dt_x0 + f_x1*d2dt_x1)

    return hessian_matrix

def partial_trace(matrixA_B, A= True, B=False):
    
    sub = np.zeros((2,2))
    if A == True and B == False: 
        sub[0][0] = matrixA_B[0][0] + matrixA_B[1][1]
        sub[0][1] = matrixA_B[0][2] + matrixA_B[1][3]
        sub[1][0] = matrixA_B[2][0] + matrixA_B[3][1]
        sub[1][1] = matrixA_B[2][2] + matrixA_B[3][3]
        return sub
    elif A == False and B == True: 
        sub[0][0] = matrixA_B[0][0] + matrixA_B[2][2]
        sub[0][1] = matrixA_B[0][1] + matrixA_B[2][3]
        sub[1][0] = matrixA_B[1][0] + matrixA_B[3][2]
        sub[1][1] = matrixA_B[1][1] + matrixA_B[3][3]
        
        return sub
    else : 
        print('choose beetwen A and B not both')

def get_theta_inv(p: float, v: float, sigma_p: float, sigma_v: float):
    '''
    Function to calculate the linear fucntion but by inverting the fucntion
    v : The visibility
    '''
    theta = np.arccos((2*p-1)/(v))

    dt_dp = -2 / (v * np.sqrt(1 - ((2*p - 1)/v)**2))
    dt_dv = (2*v -1)/(v**2*np.sqrt(1 - ((2*p - 1)/v)**2))
    theta_error = np.sqrt((dt_dp*sigma_p)**2+(dt_dv*sigma_v)**2)

    return (theta, theta_error)
def std(Theta : float, Theta_est:list):
    return(np.sqrt((1/len(Theta_est))*sum((Theta_esta - Theta)**2 for Theta_esta in Theta_est)))

def r_z(x):
    unitary = np.array([[np.exp(-1j*x/2), 0],
                        [0, np.exp(1j*x/2)]])
    return unitary

### Defining the functions that will define the bounds
def sum_min_bound(x):
    return np.sum(x)

def sum_up_bound(x):
    return (np.pi)-np.sum(x)


class Estimator:
    def __init__(self, state, qubit, meas_stats, initial_guess, bnds, function_type = "visibility", *args):
        """
        function_type: Selects which internal function to use ('visibility' or 'density_matrix').
        """
        self.state = state
        self.qubit = qubit
        self.meas_stats = meas_stats
        self.x0 = initial_guess
        self.function_type = function_type  # Store which function to use
        self.args = args[0]
        self.bnds = bnds
        self.set_function_type(self.function_type, self.args)

    def likelihood_function_v(self, x):
        """
            Likelihood function for a given array of theta values [x0, x1, x2, x3].
            x: np.ndarray
                Array of theta values [x0, x1, x2, x3].
        """            
        v = self.visibility #Here it should be [0] but when I use set_function_type, it shouldn't
        prob_distribution = 0
        
        # Replace theta with x (array)
        for outcome in self.meas_results:
            prob_distribution -= np.log((1 - (-1)**outcome * v * np.cos(np.sum(x))) / 2)

        return prob_distribution  # Negative log-likelihood

    def likelihood_function_plus_v(self, x):
        """
            Likelihood function for a given array of theta values [x0, x1, x2, x3].
            x: np.ndarray
                Array of theta values [x0, x1, x2, x3].
        """            
        v1, v2, v3, v4 = self.visibility #Here it should be [0] but when I use set_function_type, it shouldn't
        prob_distribution = 0
        
        # Replace theta with x (array)
        for outcome in self.meas_results:
            prob_distribution -= np.log((1 - (-1)**outcome * v1*v2*v3*v4*np.cos(x[0])*np.cos(x[1])*np.cos(x[2])*np.cos(x[3])) / 2)

        return prob_distribution  # Negative log-likelihood

    def likelihood_function_Bell_v(self, x):
        """
            Likelihood function for a given array of theta values [x0, x1, x2, x3].
            x: np.ndarray
                Array of theta values [x0, x1, x2, x3].
        """            
        v1, v2 = self.visibility #Here it should be [0] but when I use set_function_type, it shouldn't
        prob_distribution = 0
        
        # Replace theta with x (array)
        for outcome in self.meas_results:
            prob_distribution -= np.log((1 - (-1)**outcome * v1 * v2 * np.cos(x[0])* np.cos(x[1])) / 2)

        return prob_distribution  # Negative log-likelihood

    def likelihood_function_dm(self, x):
        """
        Likelihood function for a given array of theta values [x0, x1, x2, x3].
        x: np.ndarray
            Array of theta values [x0, x1, x2, x3].
        """
        dm = self.dm
        hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)
        diag = [[1, 0],[0, 1]]
        U1 = ft.reduce(np.kron, [r_z(theta) for theta in x])
        U2 = ft.reduce(np.kron, [hadamard] * len(x))
        U = U2@U1
        pre_meas_dm = U@dm@np.conjugate(np.transpose(U))

        if self.state == 'GHZ' and len(x) == 4:
            meas_dm = pre_meas_dm
        
        elif self.state == 'Bell' and len(x) == 2:
            Id = np.kron(diag, diag)
            if self.qubit == 0 :
                meas_dm = np.kron(pre_meas_dm, Id)
            if self.qubit == 1 :
                meas_dm = np.kron(Id, pre_meas_dm)
            
        elif self.state == 'Plus' and len(x) == 1:
            Id = diag
            if self.qubit == 0 :
                meas_dm = ft.reduce(np.kron,(pre_meas_dm, Id, Id, Id))
            if self.qubit == 1 :
                meas_dm = ft.reduce(np.kron,(Id, pre_meas_dm, Id, Id))
            if self.qubit == 2 :
                meas_dm = ft.reduce(np.kron,(Id, Id, pre_meas_dm, Id))
            if self.qubit == 3 :
                meas_dm = ft.reduce(np.kron,(Id, Id, Id, pre_meas_dm))
        else:
            raise ValueError("Invalid State. Choose GHZ, Bell_Qubit. Be Aware of the length of your initial guess.")
        
        prob_distribution = 0
        for outcome in self.meas_results:
            prob_distribution -= np.log(np.real(outcome@meas_dm@outcome))
        
        return prob_distribution
    
    def likelihood_function_dm_attack(self, x):
        """
        Likelihood function for a given array of theta values [x0, x1, x2, x3].
        x: np.ndarray
            Array of theta values [x0, x1, x2, x3].
        """
        if len(self.state) == 2 :
            dm_classic,dm_attack,self.nu = self.args_attack
            hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)
            U1 = ft.reduce(np.kron, [r_z(theta) for theta in x])
            U2 = ft.reduce(np.kron, [hadamard] * len(x))
            U = U2@U1
            pre_meas_dm_attack = U@dm_attack@np.conjugate(np.transpose(U))
            pre_meas_dm_classic = U@dm_classic@np.conjugate(np.transpose(U))
            prob_distribution = 0

            for i,outcome in enumerate(self.meas_results):
                if self.nu[i] == 0 :
                    meas_dm = pre_meas_dm_classic
                if self.nu[i] == 1 : 
                    meas_dm = pre_meas_dm_attack

                prob_distribution -= np.log(np.real(outcome@meas_dm@outcome))
            
            return prob_distribution
        else : 
            print("It's an attack scenario you should mention two states and everything should be in the same dimension as for a GHZ ")
        
    def likelihood_function_dm_attack_2_states(self, x):
        """
        Likelihood function for a given array of theta values [x0, x1, x2, x3].
        x: np.ndarray
            Array of theta values [x0, x1, x2, x3].
        """
        if len(self.state) == 3 :
            dm_classic,dm_attack,dm_attack2,self.nu = self.args_attack
            hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)
            U1 = ft.reduce(np.kron, [r_z(theta) for theta in x])
            U2 = ft.reduce(np.kron, [hadamard] * len(x))
            U = U2@U1
            pre_meas_dm_attack = U@dm_attack@np.conjugate(np.transpose(U))
            pre_meas_dm_classic = U@dm_classic@np.conjugate(np.transpose(U))
            pre_meas_dm_attack2 = U@dm_attack2@np.conjugate(np.transpose(U))
            prob_distribution = 0

            for i,outcome in enumerate(self.meas_results):
                
                if self.nu[i] == 0 :
                    meas_dm = pre_meas_dm_classic
                if self.nu[i] == 1 : 
                    meas_dm = pre_meas_dm_attack
                if self.nu[i] == 2 : 
                    meas_dm = pre_meas_dm_attack2

                prob_distribution -= np.log(np.real(outcome@meas_dm@outcome))
            
            return prob_distribution
        else : 
            print("It's an attack scenario you should mention three states and everything should be in the same dimension as for a GHZ ")
        
    
    def get_theta(self, get_cov = False, set_lim = None):      
        if set_lim != None:
            self.set_counts_array_limit(set_lim)

        cons = [{'type': 'ineq', 'fun': sum_min_bound},
                {'type': 'ineq', 'fun': sum_up_bound}]

        mle = []
        mle_local_min_fun = []
        for i, x0 in enumerate(self.x0):
            mle.append(sp.minimize(self.likelihood_function,
                                    x0,
                                    method = 'SLSQP',
                                    bounds = self.bnds,
                                    constraints = cons))
            
            mle_local_min_fun.append(mle[i].fun)
        
        self.mle = mle[mle_local_min_fun.index(min(mle_local_min_fun))]
        if get_cov is True:
            self.get_cov_matrix()
        
        return self.mle
    
    def get_cov_matrix(self):
        hessian_func = nd.Hessian(self.likelihood_function)  # Create a Hessian function
        self.cov_matrix = hessian_func(self.mle.x)  # Evaluate the Hessian

        return self.cov_matrix

    def get_variance(self, a_vector):
        a_vector_norm = a_vector/np.linalg.norm(a_vector)
        self.fisher_bound = a_vector_norm@self.cov_matrix@a_vector_norm
        var2_norm = 1/self.fisher_bound
        var2 = (np.linalg.norm(a_vector)**2)*var2_norm
        self.var = np.sqrt(var2)

        return self.var
    
    def get_variance2(self, a_vector):
        a_vector_norm = a_vector/np.linalg.norm(a_vector)
        self.fisher_bound = a_vector_norm@self.cov_matrix@a_vector_norm
        var2_norm = 1/self.fisher_bound
        var2 = (np.linalg.norm(a_vector)**2)*var2_norm
        self.var2 = var2

        return self.var2

    def set_function_type(self, function_type, *args):
        self.function_type = function_type
        """Choose which function to use and perform optimization."""
        # Select function based on user choice
        if self.function_type == "visibility":
            self.likelihood_function = self.likelihood_function_v
            self.meas_results = self.meas_results = np.sum((self.meas_stats.counts_array*(self.meas_stats.eigenvalues + 1)/2), axis = 1)
            self.visibility = args[0]
        elif self.function_type == "density_matrix":
            self.likelihood_function = self.likelihood_function_dm
            self.meas_results = self.meas_stats.counts_array
            self.dm = args[0]
        elif self.function_type == "visibility_Bell":
            self.likelihood_function = self.likelihood_function_Bell_v
            self.meas_results = self.meas_results = np.sum((self.meas_stats.counts_array*(self.meas_stats.eigenvalues + 1)/2), axis = 1)
            self.visibility = args[0]
        elif self.function_type == "visibility_plus":
            self.likelihood_function = self.likelihood_function_plus_v
            self.meas_results = self.meas_results = np.sum((self.meas_stats.counts_array*(self.meas_stats.eigenvalues + 1)/2), axis = 1)
            self.visibility = args[0]
        elif self.function_type == "attack":
            self.likelihood_function = self.likelihood_function_dm_attack
            self.meas_results = self.meas_stats.counts_array
            self.args_attack = args[0]
        elif self.function_type == "attack_2_states":
            self.likelihood_function = self.likelihood_function_dm_attack_2_states
            self.meas_results = self.meas_stats.counts_array
            self.args_attack = args[0]
        else:
            raise ValueError("Invalid function type. Choose 'visibility' or 'density_matrix'. If 'attack' chosen the args should be order like : dm_classic, dm_attack, nu")

    def set_counts_array_limit(self, lim):
        self.lim = lim
        self.meas_results = self.meas_results[:self.lim]
    

