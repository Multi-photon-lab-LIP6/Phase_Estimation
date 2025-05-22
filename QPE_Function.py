import numpy as np
import itertools
import functools as ft 
import random
from scipy.optimize import root_scalar,fsolve,minimize
import math


def Parity(Y:float,V:float,N:int):
    
    '''
    Function to claculate the parity probability P_even, P_odd
    Y : The linear fucntion of thetas
    V : The visibility
    N : The number of measurment (Here the number of qubits)
    '''
    return((1 + V*np.cos(N*Y))/(2),(1 + -V*np.cos(N*Y))/(2))

def Maximun_Likelihood(Y:float,V:float,N:int,X):
    
    '''-
    Function to calculate the maximun likelihood function of the parity distribution
    Y : The linear fucntion of thetas
    V : The visibility
    N : The number of measurment (Here the number of qubits)
    X : The measurement result. 
    '''

    Dis_prob = (1 + V*np.cos(N*Y-X*np.pi))/(2)
    return(-np.sum(np.log(Dis_prob)))

def Prob_plus(V:float,N:int,P:float):
    '''
    Function to calculate the linear fucntion but by inverting the fucntion
    V : The visibility
    Meax_X : Result from the XXXX Measurement
    '''
    return((np.arccos((2*P-1)/(V))/N))


def Prob_minus(V:float,N:int,P:float):
    return((np.arccos((-2*P+1)/(V))/N))
 
def N_rob(epsilon,r,lamb,nu,delta):
    return ((((np.log(1/delta))+4*lamb*nu**2)/(epsilon))*(((lamb * np.sqrt(2 * (nu)) + 1)/(lamb * (nu) * (1 - r)))**2))

def failure(epsilon, lamb, nu, r):
    """
    Original failure function.
    """
    return (((lamb * math.sqrt(2 * nu) + r) / (lamb * math.sqrt(2 * nu) + 1)) * nu * epsilon)

def find_epsilon(failure_value, lamb, nu, r):
    """
    Find epsilon numerically by solving the equation failure(epsilon) = failure_value.
    
    Parameters:
        failure_value (float): The target value of the failure function.
        lamb (float): Parameter lambda.
        nu (float): Parameter nu.
        r (float): Parameter r.
    
    Returns:
        float: The calculated epsilon.
    """
    # Define the equation to solve: failure(epsilon) - failure_value = 0
    def equation(epsilon):
        return failure(epsilon, lamb, nu, r) - failure_value
    
    # Use a root-finding method to solve for epsilon
    solution = root_scalar(equation, bracket=[0, 1], method='brentq')  # Bracket defines the range of epsilon
    
    if not solution.converged:
        raise ValueError("Failed to find a solution for epsilon.")
    return solution.root


def sigificance(epsilon,r,lamb,nu):
    return (lambda x : N_rob(epsilon,r,lamb,nu,x))

def get_sigificance(epsilon,r,lamb,nu):
    
    func = sigificance(epsilon,r,lamb,nu)
    intial_guess = 0.5
    sigificance_sol = fsolve(func,intial_guess)
    return(sigificance_sol)
    
def Maximun_Likelihood(Y:float,V:float,N:int,X):
    
    '''
    Function to calculate the maximun likelihood function of the parity distribution
    Y : The linear fucntion of thetas
    V : The visibility
    N : The number of measurment (Here the number of qubits)
    X : The measurement result. 
    '''

    Dis_prob = (1 + V*np.cos(N*Y-X*np.pi))/(16)
    return(-np.sum(np.log(Dis_prob)))

def find_min_delta(failure_rate,N, epsilon, lamb, nu):
    # Define the objective function
    def objective(delta, r):
        return abs(N_rob(epsilon, r, lamb, nu, delta) - N)  # Minimize the error to target N
    
    def cons(r):
        return failure(epsilon, lamb, nu, r) - failure_rate
    
    # Bounds: 0 < delta < 1
    bounds = [(1e-12, 1 - 1e-12)]  # Avoid invalid log(0) values

    # Initial guess for delta
    initial_delta = 0.5
    initial_r = 0.5
    cons = [{'type':'eq', 'fun': cons}]
    # Minimize the objective function
    result = minimize(objective, x0=[initial_delta, initial_r], bounds=bounds, constraints = cons, method='L-BFGS-B')
    

    return result.x[0]  # Return the delta that minimizes the objective

def last_4chars(x):
    return(int((x.split("_"))[2].split(".")[0]))

def last_2chars(x):
    return(int((x.split("_"))[1].split(".")[0]))

def separation(measurement_basis,s):  
    compt = min(s)
    z = 0
    y = 0
    measurement_basis_total =[]
    for i in measurement_basis:    
            if int(last_4chars(i)) == compt :
                z = z + 1
            else : 
                compt = compt + 1
                measurement_basis_total.append(measurement_basis[y:z])
                y = z
                z = z+1
    measurement_basis_total.append(measurement_basis[y:z])
    return measurement_basis_total

def calculate_epsilon(failure, lamb, nu, r):
    """
    Calcule epsilon en fonction de failure, lamb, nu et r.
    
    Parameters:
    - failure : la valeur du taux de défaillance.
    - lamb : paramètre lambda.
    - nu : paramètre nu.
    - r : paramètre r.

    Returns:
    - epsilon : la valeur de epsilon calculée.
    """
    # Calcul du facteur de proportionnalité
    factor = (lamb * math.sqrt(2 * nu) + 1) / (lamb * math.sqrt(2 * nu) + r)
    
    # Calcul de epsilon
    epsilon = (failure / nu) * factor
    return epsilon

def calculate_delta(N_rob, epsilon, r, lamb, nu):
    """
    Calcule delta en fonction de N_rob, epsilon, r, lamb et nu.
    
    Parameters:
    - N_rob : la valeur de N_rob.
    - epsilon : la valeur de epsilon.
    - r : paramètre r.
    - lamb : paramètre lambda.
    - nu : paramètre nu.
    
    Returns:
    - delta : la valeur de delta calculée.
    """
    # Calcul du facteur qui dépend des paramètres
    factor = (lamb * np.sqrt(2 * nu) + 1) / (lamb * nu * (1 - r))
    factor_squared = factor ** 2
    
    # Calcul de la valeur de delta
    delta = 1 / np.exp((N_rob * epsilon) / factor_squared - 4 * lamb * nu**2)
    
    return delta

def calculate_delta(N_rob, epsilon, r, lamb, nu):
    """
    Calcule delta en fonction de N_rob, epsilon, r, lamb et nu.
    
    Parameters:
    - N_rob : la valeur de N_rob.
    - epsilon : la valeur de epsilon.
    - r : paramètre r.
    - lamb : paramètre lambda.
    - nu : paramètre nu.
    
    Returns:
    - delta : la valeur de delta calculée.
    """
    # Calcul du facteur qui dépend des paramètres
    factor = (lamb * np.sqrt(2 * nu) + 1) / (lamb * nu * (1 - r))
    factor_squared = factor ** 2
    
    # Calcul de la valeur de delta
    delta = 1 / np.exp((N_rob * epsilon) / factor_squared - 4 * lamb * nu**2)
    
    return delta

def calculate_r(N_rob, epsilon, delta, lamb, nu):
    """
    Calcule r en fonction de N_rob, epsilon, delta, lamb et nu.

    Parameters:
    - N_rob : la valeur de N_rob.
    - epsilon : la valeur de epsilon.
    - delta : la valeur de delta.
    - lamb : paramètre lambda.
    - nu : paramètre nu.

    Returns:
    - r : la valeur de r calculée.
    """
    # Calcul du facteur qui dépend des paramètres
    factor = (lamb * np.sqrt(2 * nu) + 1) / (lamb * nu)
    factor_squared = factor ** 2

    # Calcul de r à partir de l'équation donnée
    exponent_term = (N_rob * epsilon) / factor_squared - 4 * lamb * nu**2
    r = 1 - (np.log(1 / delta) / exponent_term)

    return r


def Prob(P:float,N:int,V:float,signe:bool):
    
    '''
    Function to calculate the linear fucntion but by inverting the fucntion
    V : The visibility
    Meax_X : Result from the XXXX Measurement
    '''
    
    if (signe) :
        return((np.arccos((2*P-1)/(V))/N))
    else :
        return((np.arccos((-2*P+1)/(V))/N))
    
def STD(Theta : float, Theta_est:list):
    return(np.sqrt((1/len(Theta_est))*sum((Theta_esta - Theta)**2 for Theta_esta in Theta_est)))

def derivate_theta_P(P,F,N):
    return -2/(N*F*np.sqrt(1-((2*P-1)**2)/F**2))

def derivate_theta_F(P,F,N):
    return (2*P-1)/(N*(F**2)*np.sqrt(1-((2*P-1)**2)/F**2))


def fisher(V,theta):
    return(((V**2)*(4**2)*(np.sin(4*theta)**2))/(1-(V**2)*(np.cos(4*theta)**2)))