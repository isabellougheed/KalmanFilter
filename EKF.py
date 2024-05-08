# Import packages
import numpy as np
import control
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import linalg
from MSD import MSD_NL_System

"""
This file is to set up the Extended Kalman filter for the mass spring damper system

All plotting was done in this file.

"""

class EKF:
    def __init__(self, A, B, L, C, D, Q, R):
        self.A = A
        self.B = B
        self.L = L
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        # I'll initialize these after discretization
        self.A_d = 0
        self.B_d = 0
        self.Q_d = 0
        self.R_d = 0
        self.e_tol = 10**(-3) # IS THIS GOOD??

    def discretize(self,T):
        # building large matrix
        xi = np.block([[A, L@(Q*(L.T)), np.zeros((2,2)), np.zeros((2,1))], [np.zeros((2,2)), -A.T, np.zeros((2,2)), np.zeros((2,1))], [np.zeros((2,2)), np.zeros((2,2)), A, B], [np.zeros((1,2)),np.zeros((1,2)),np.zeros((1,2)),np.zeros((1,1))]])
        upsilon = linalg.expm(xi*T) 

        # extract A_d, B_d, Q_d
        upsilon_11 = upsilon[0:2, 0:2]
        upsilon_12 = upsilon[0:2, 2:4]
        upsilon_34 = upsilon[4:6, 6]
        A_d = upsilon_11
        B_d = np.array(upsilon_34) #it transposes for some reason
        B_d = np.array([[upsilon_34[0]], [upsilon_34[1]]])
        Q_d = upsilon_12@(upsilon_11.T)
        R_d = R/T

        #initialize parameters
        self.A_d = A_d
        self.B_d = B_d
        self.Q_d = Q_d
        self.R_d = R_d
        return A_d, B_d, Q_d, R_d
    
    def c_k(self, msd, x):
        """
        This method evaluates the jacobian of g(x) in order to linearize and discretize C about different states x
        """
        # Check numerically!!!
        x_1 = x[0]
        d = msd.d
        h = msd.h
        return np.array([(d + x_1)/(np.sqrt((d + x_1)**2 + h**2))])

    
    def predict(self, x_prior, P_prior, u):
        xk_p = A_d @ x_prior + B_d*u
        Pk_p = A_d @ P_prior @ A_d.T + Q_d
        return xk_p, Pk_p
    
    def correct_ekf(self, msd, xk_p, Pk_p, y):
        C_k = self.c_k(msd, xk_p)
        K_k = Pk_p @ (C_k.reshape(1, -1)).T *(C_k @ Pk_p @ (C_k.reshape(1, -1)).T + R_d)**(-1)
        x_k = xk_p + K_k *(y - msd.g(xk_p)) 
        #P_k = (np.eye(2,2) - K_k @ C_k.reshape(1, -1)) @ Pk_p @(np.eye(2,2) - K_k @ C_k.reshape(1, -1)).T + K_k*R_d*K_k.T
        ### in slides, he just does this... TRY BOTH
        P_k = (np.eye(2,2) - K_k @ C_k.reshape(1, -1)) @ Pk_p
        return K_k, x_k, P_k

    def correct_iekf(self, msd, xk_p, Pk_p, y):
        counter = 0
        # This is the first iteration
        x_k_j = xk_p # to start
        C_k = self.c_k(msd, x_k_j)
        K_k_j = Pk_p @ (C_k.reshape(1, -1)).T *(C_k @ Pk_p @ (C_k.reshape(1, -1)).T + R_d)**(-1)
        x_k_j_1 = xk_p + K_k_j *(y - msd.g(x_k_j)- C_k @ (xk_p - x_k_j))
        P_k_j_1 = (np.eye(2,2) - K_k_j @ C_k.reshape(1, -1)) @ Pk_p
        counter += 1

        while np.linalg.norm(x_k_j_1 - x_k_j) > self.e_tol:
            x_k_j = x_k_j_1
            # or max number of iterations maybe??
            C_k = self.c_k(msd, x_k_j)
            K_k_j = Pk_p @ (C_k.reshape(1, -1)).T *(C_k @ Pk_p @ (C_k.reshape(1, -1)).T + R_d)**(-1)
            x_k_j_1 = xk_p + K_k_j *(y - msd.g(x_k_j)- C_k @ (xk_p - x_k_j))
            P_k_j_1 = (np.eye(2,2) - K_k_j @ C_k.reshape(1, -1)) @ Pk_p
            counter += 1

        return K_k_j, x_k_j_1, P_k_j_1, counter
    
    def filter(self, msd, a_n, y_n, x_hat0, P_hat0, steps, frequ_a, frequ_y, isIEKF):
        # Making a list to add all estimated states and covariances at each step into
        x_hat_k = []
        P_hat_k = []
        sigma3_k = []
        iterations = []

       # initializing lists with initial guesses
        x_hat_k.append(x_hat0.flatten())
        P_hat_k.append(P_hat0)
        sigma3_k.append(3*np.sqrt(np.diag(P_hat0))) 

        #Setting up x_k-1, P_k-1
        x_k_1 = x_hat0
        P_k_1 = P_hat0

        for i in range(0, steps-1):
            # predict
            xk_p, Pk_p = self.predict(x_k_1, P_k_1, a_n[i])

            # correct
            if isIEKF:
                K_k, x_k, P_k, counter = self.correct_iekf(msd, xk_p, Pk_p, y_n[i])
                iterations.append(counter)
            else:
                K_k, x_k, P_k = self.correct_ekf(msd, xk_p, Pk_p, y_n[i])

            #### ADD SECTION FOR IF DIFFERENT MEASURING FREQUENCIES

            # next iteration
            x_k_1 = x_k
            P_k_1 = P_k
        
            # add to arrays
            x_hat_k.append(x_k.flatten())
            P_hat_k.append(P_k)
            S = np.diag(P_k) 
            sigma3_k.append(3*np.sqrt(S))
            
        return x_hat_k, P_hat_k, sigma3_k, iterations




#%%
#set up system
m = 2.5
k = 30
c = 3
A = 1
w = 5
h = 1
d = 1
sys = MSD_NL_System(m,c,k,A,w,h,d)
x0 = np.array([[5], [0]])

# Time
dt = 1e-3
t_start = 0
t_end = 10
t = np.arange(t_start, t_end, dt) # t has 10000 steps

#%%
# GENERATING GROUND TRUTH FOR NO INPUT
sol = integrate.solve_ivp(
    sys.f1,
    (t_start, t_end),
    x0.ravel(),
    t_eval=t,
    rtol=1e-6,
    atol=1e-6,
    method='RK45',
) #I like this better because it returns all of the solution steps and you don't need to iterate through
t_sol = sol.t
x_sol = sol.y
r_sol = x_sol[0]
dotr_sol = x_sol[1]
a_sol = (1/m)*( - k*r_sol - c*dotr_sol)

#%%
# DISCRETIZATION

# rewrite system with noise
A = np.array([[0,1],[0,0]])
B = np.array([[0],[1]])
L = np.array([[0],[1]])
C = np.array([d/(np.sqrt(d**2 + h**2)), 0])
D = 0  
Q = 0.01  # acceleration noise variance
R = 0.001 # position noise variance
ekf = EKF(A,B,L,C,D,Q,R)

# Discretize system
T = dt 
A_d, B_d, Q_d, R_d = ekf.discretize(T)

#%%
# GENERATING SENSOR MEASUREMENTS

#%%
# EXTENDED KALMAN FILTER

#%%
# PLOT ESTIMATED AND TRUE STATES

# Plotting parameters
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc("text", usetex=True)


# %%
# PLOT ERROR

# %%
# NEES TEST