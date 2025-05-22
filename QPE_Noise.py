### trying to do it as a Class for training perspective

# Function to simulate the noise and apply it to a density matrix

import numpy as np
import itertools
import scipy as scipy
import functools as ft 


def apply_angle(rho,angle):
    
    U1 = np.matrix([[np.exp(-1j*angle[0]/2),0],[0,np.exp(1j*angle[0]/2)]])
    U2 = np.matrix([[np.exp(-1j*angle[1]/2),0],[0,np.exp(1j*angle[1]/2)]])
    U3 = np.matrix([[np.exp(-1j*angle[2]/2),0],[0,np.exp(1j*angle[2]/2)]])
    U4 = np.matrix([[np.exp(-1j*angle[3]/2),0],[0,np.exp(1j*angle[3]/2)]])
    U = ft.reduce(np.kron,(U1,U2,U3,U4))
    
    return U@rho@(np.transpose(np.conjugate(U)))

def apply_angle1(rho,angle):
    angles = sum(angle)
    U1 = np.matrix([[np.exp(-1j*angles/2),0],[0,np.exp(1j*angles/2)]])
    U2 = np.identity(2)
    U3 = np.identity(2)
    U4 = np.identity(2)
    U = ft.reduce(np.kron,(U1,U2,U3,U4))
    
    return np.matmul(np.matmul(U, rho), np.conjugate(np.transpose(U)))
 


def apply_noise(rho, d:int, K:list, Nb_noise:int):
    '''
    Parameters
    ----------
    rho : Numpy 2d-array
        Density matrix of the state to apply .
    d : int
        dimension.
    K : list
        List of Kraus operators.
    Nb_noise : int
        Number of type of noise you want to apply

    Returns
    -------
    s : Numpy 2d-array
        Density matrix after noise.
    '''
    for j in range(Nb_noise):
        s = np.zeros((2**d, 2**d))
        for i in range(len(K[j])):
            s = s + np.matmul(np.matmul(K[j][i], rho), np.conjugate(np.transpose(K[j][i])))
        rho = s
    s = s
    return s

def fid(dm, target):
    '''
    Parameters
    ----------
    dm : Numpy 2d-array
        Density matrix of the state yoi have .
    target : Numpy 2d-array
        Density matrix of the state you want .
    Returns
    -------
    the fidelidy calculate in different ways if the shape of your matrice is not the same as your target
    '''
    shape=np.shape(target)

    if (len(shape)>1): 
        return (np.trace(scipy.linalg.sqrtm(scipy.linalg.sqrtm(target)@dm@scipy.linalg.sqrtm(target)))**2)
    else:
        return np.transpose(np.conjugate(target))@dm@target



class Noise:
    """
    Class for all the noise that could be created depending on the size of our state
    """
    def __init__(self, state):
            '''
        Parameters
        ----------
        rho : Numpy 2d-array
            Density matrix of the state to apply .
            '''
            self.state = state

    @property
    
    def qbit_number(self):
        
        return np.log2(self.state.shape[0]).astype(int)

    def depolarising_Kraus_operators(self, eta, verify:bool=False):
        # Nielsen & Chuang, page 378.
        '''
        Parameters
        ----------
        eta : float
            noise parameter:
                eta=1: pure state and no noise,
                eta=0: only noise, the state is lost completely.
        verify : bool, optional
            Verification for the Kraus operators. The default is False.

        Returns
        -------
        K : TYPE
            List of amplitude damping Kraus operators.
        '''
        
        a = np.sqrt(1-3*eta/4)
        b = np.sqrt(eta/4)
        
        K_A = a * np.array([[1, 0], [0, 1]])
        K_B = b * np.array([[0, 1], [1, 0]])
        K_C = b * np.array([[0, -1j], [1j, 0]])
        K_D = b * np.array([[1, 0], [0, -1]])
        
        lists = list(itertools.product([K_A, K_B, K_C, K_D], repeat=self.qbit_number))
        
        K = []
        for i in range(len(lists)):
            k = np.array([1])
            for j in range(len(lists[i])):
                k = np.kron(k, lists[i][j])
            
            K.append(k)
            
        if verify:
            verify_K_One(K)
        
        return K


    def dephasing_Kraus_operators(self, eta, verify:bool=False):
        # Majid
        
        K_A = np.sqrt((1-eta)) * np.array([[1, 0], [0, 1]])
        K_B = np.sqrt(eta) * np.array([[1, 0], [0, -1]])
        
        lists = list(itertools.product([K_A, K_B], repeat=self.qbit_number))
        
        K = []
        for i in range(len(lists)):
            k = np.array([1])
            for j in range(len(lists[i])):
                k = np.kron(k, lists[i][j])
            
            K.append(k)
            
        if verify:
            verify_K_One(K)
        
        return K

    def amplitude_damping_Kraus_operators(self,eta:float, verify:bool=False):
        # Nielsen & Chuang, page 380.
        '''
        Parameters
        ----------

        eta : float
            noise parameter:
                eta=1: pure state and no noise,
                eta=0: only noise, the state is lost completely.
        verify : bool, optional
            Verification for the Kraus operators. The default is False.

        Returns
        -------
        K : list
            List of amplitude damping Kraus operators.
        '''
        
        K_A = np.array([[1, 0], [0, np.sqrt(1-eta)]])
        K_B = np.array([[0, np.sqrt(eta)], [0, 0]])
        
        lists = list(itertools.product([K_A, K_B], repeat=self.qbit_number))
        
        K = []
        for i in range(len(lists)):
            k = np.array([1])
            for j in range(len(lists[i])):
                k = np.kron(k, lists[i][j])
            
            K.append(k)
        
        if verify:
            verify_K_One(K)
        
        return K

    def phase_damping_Kraus_operators(self,eta:float, verify:bool=False):
        # Nielsen & Chuang, page 383.
        '''
        Parameters
        ----------
        eta : float
            noise parameter:
                eta=1: pure state and no noise,
                eta=0: only noise, the state is lost completely.
        verify : bool, optional
            Verification for the Kraus operators. The default is False.

        Returns
        -------
        K : list
            List of phase damping Kraus operators.
        '''
        
        K_A = np.array([[1, 0], [0, np.sqrt(1-eta)]])
        K_B = np.array([[0, 0], [0, np.sqrt(eta)]])
        
        lists = list(itertools.product([K_A, K_B], repeat=self.qbit_number))
        
        K = []
        for i in range(len(lists)):
            k = np.array([1])
            for j in range(len(lists[i])):
                k = np.kron(k, lists[i][j])
            
            K.append(k)
        
        if verify:
            verify_K_One(K)
        
        return K

    def verify_K(self,K:list,Nb_noise:int, verbose:bool=True):
        '''
        Parameters
        ----------
        K : list
            List of Kraus operators.
        d : int
            Dimension (i.e. number of parameters).
        verbose : bool, optional
            If True, it prints the result of the matrix multiplication of (K^t)*K,
            which should be the dxd identity matrix.
            The default is False.

        Returns
        -------
        bool
        True, if the Kraus operators are well defined.
        False, otherwise.
        '''
        
        s = np.zeros((2**self.qbit_number, 2**self.qbit_number))
        print('')
        for j in range(Nb_noise):
            for i in range(len(K[j])):
                s = s + np.matmul(np.conjugate(np.transpose(K[j][i])), K[j][i])
                
                #print(i, s) # to print step by step.
        if verbose:
            print('\nVerification:\n', np.real(s/4), '\n', 50*'-')
            
    def verify_K_One(self,K:list,verbose:bool=True):
    
        s = np.zeros((2**self.qbit_number, 2**self.qbit_number))
        print('')
        for i in range(len(K)):
            s = s + np.matmul(np.conjugate(np.transpose(K[i])), K[i])
            
            #print(i, s) # to print step by step.
        if verbose:
            print('\nVerification:\n', np.real(s), '\n', 50*'-')

class QPE:
    
    """
    Class for all the noise that could be created depending on the size of our state
    """
    def __init__(self, state):
            '''
        Parameters
        ----------
        rho : Numpy 2d-array
            Density matrix of the state to apply .
            '''
            self.state = state

    @property
    
    def qbit_number(self):
        
        return np.log2(self.state.shape[0]).astype(int)
    
    def Failure_rate(self, List_Meas : list):
        
        return((1 - 1/5*(np.sum([np.trace(self.state@List_Meas[i]) for i in range(len(List_Meas))])))/2)
    
    
