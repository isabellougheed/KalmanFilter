# Import packages
import numpy as np
import control
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import linalg
from scipy import stats
from MSD import MSD_System
import random

"""
This file is to set up the Kalman filter for the mass spring damper system set up in MSD.py

All plotting was done in this file.

"""

class Kalman_Filter:
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

    def discretize(self,T):
        """
        This method discretizes a state space system using Steven Dahdah's method.
        """
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
    
    def predict(self, x_prior, P_prior, a):
        """
        This method takes a prior state estimation, a prior covariance, and an acceleration measurement 
        and returns the next predicted state and covariance.

        x_k+1 = (A_k)(x_k) + (B_k)(u_k)

        P_k+1 = (A_k)(P_k)(A_k).T + Q_k
        """
        xk_p = A_d @ x_prior + B_d*a #x_k+1 = (A_k)(x_k) + (B_k)(u_k)
        Pk_p = A_d @ P_prior @ A_d.T + Q_d  #P_k+1 = (A_k)(P_k)(A_k).T + Q_k
        return xk_p, Pk_p
    
    """
    DO CORRECT METHOD HERE!!
    """
    
    def filter(self, a_n, y_n, x_hat0, P_hat0, steps, frequ_a, frequ_y):
        """ 
        This method represents the Kalman filter as whole. It predicts and corrects over all time steps.
        
        This method returns estimated states, covariances, and the sigma3 bounds at each time step.
        """
        # Making a list to add all estimated states and covariances at each step into
        x_hat_k = []
        P_hat_k = []
        sigma3_k = []


        # initializing lists with initial guesses
        x_hat_k.append(x_hat0.flatten())
        P_hat_k.append(P_hat0)
        U0, S0, V0 = np.linalg.svd(P_hat0)
        sigma3_k.append(S0)
        # FIX THIS !!!!

        #Setting up x_k-1, P_k-1
        x_k_1 = x_hat0
        P_k_1 = P_hat0

        for i in range(0, steps-1):
            # prediction
            xk_p, Pk_p = self.predict(x_k_1, P_k_1, a_n[i])

            # correction
            K_k = Pk_p @ (C.reshape(1, -1)).T *(C @ Pk_p @ (C.reshape(1, -1)).T + R_d)**(-1) 

            if frequ_a != frequ_y:
                # if sampling frequencies not the same, can't do correction every step because missing y measurements
                if i % (frequ_a/frequ_y) == 0:
                    # if this step has a y measurement, can do correction
                    x_k = xk_p + K_k *(y_n[i] - C @ xk_p) #x_k+1|k+1 = x_k+1|k + K_k+1(y_k+1 - (C_k+1)(x_k+1|k))
                    P_k = (np.eye(2,2) - K_k @ C.reshape(1, -1)) @ Pk_p @(np.eye(2,2) - K_k @ C.reshape(1, -1)).T + K_k*R_d*K_k.T # P_k+1|k+1 = (I - (K_k+1)(C_k+1))P_k+1|k(I - (K_k+1)(C_k+1)).T + (K_k+1)(R_k+1)(K_k+1).T
                else:
                    # skip correction 
                    x_k = xk_p
                    P_k = Pk_p
            else:
                # can do correction every time if a,y have same sampling rate
                x_k = xk_p + K_k *(y_n[i] - C @ xk_p) #x_k+1|k+1 = x_k+1|k + K_k+1(y_k+1 - (C_k+1)(x_k+1|k))
                P_k = (np.eye(2,2) - K_k @ C.reshape(1, -1)) @ Pk_p @(np.eye(2,2) - K_k @ C.reshape(1, -1)).T + K_k*R_d*K_k.T # P_k+1|k+1 = (I - (K_k+1)(C_k+1))P_k+1|k(I - (K_k+1)(C_k+1)).T + (K_k+1)(R_k+1)(K_k+1).T

            # next iteration
            x_k_1 = x_k
            P_k_1 = P_k
        
            # add to arrays
            x_hat_k.append(x_k.flatten())
            P_hat_k.append(P_k)
            S = np.diag(P_k) 
            sigma3_k.append(3*np.sqrt(S))

        return x_hat_k, P_hat_k, sigma3_k
    
    def nees(self, estimated_states, true_states, P):
        """
        This method takes an array of all estimated states at all time steps, an array of all covariance at all time steps, and an array of all true states at all time steps.

        This method returns the mahalanobis distances at each iteration in an array.
        """

        d_2 = np.zeros(len(estimated_states))
        for i in range(len(estimated_states)):
            e = estimated_states[i] - true_states[i]
            d_2[i] = (e.T)@(np.linalg.inv(P[i]))@e

        #stats.chi2.ppf(0.95)

        """
        - get a new d at each time step
        - should converge to 2 with one trajectory bc 2 dof
        - converge to 2 * N, num trials with monte carlo (divide, so averages out to 1)
        - random covariance and IC centered around mean
        - bounds are 2.5 and 97.5% --> 95

        - 100 trials
        - Nees over confident above condience bound, underconfident below lower boumd

        """
        return d_2

# %%
# GENERATING GROUND TRUTH DATA FOR NO INPUT
           
# Time
dt = 1e-3
t_start = 0
t_end = 10
t = np.arange(t_start, t_end, dt) # t has 10000 steps

#set up system
m = 2.5
m = 5
k = 30
c = 3
A = 1
w = 5
sys = MSD_System(m,c,k,A,w)
x0 = np.array([[5], [0]])

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
# PLOTTING TRAJECTORY WITH ZERO INPUT

# Plotting parameters
#plt.rc('lines', linewidth=2)
#plt.rc('axes', grid=True)
#plt.rc('grid', linestyle='--')
plt.rc("figure", figsize = (11.5, 8.5))
plt.rc("font", family = "Times New Roman", size = 20)
plt.rc("axes", grid = True, labelsize = 20)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("grid", linestyle = "--")
plt.rcParams["lines.markersize"] = 2

# Plot the response of x1, x2 vs. time
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$x(t)$ (units)')
# Plot data
ax.plot(t_sol, r_sol, label='$x_1(t)$', color='C0')
ax.plot(t_sol, dotr_sol, label='$x_2(t)$', color='C1')
ax.legend(loc='upper right')
#fig.tight_layout()
plt.show()

#%%
# GENERATING GROUND TRUTH FOR SINUSOIDAL INPUT
sol2 = integrate.solve_ivp(
    sys.f2,
    (t_start, t_end),
    x0.ravel(),
    t_eval=t,
    rtol=1e-6,
    atol=1e-6,
    method='RK45',
) #I like this better because it returns all of the solution steps and you don't need to iterate through
t_sol2 = sol2.t
x_sol2 = sol2.y
r_sol2 = x_sol2[0]
dotr_sol2 = x_sol2[1]
u = sys.input(t_sol2)
a_sol2 = (1/m)*(u - k*r_sol2 - c*dotr_sol2)
#%%
# PLOTTING TRAJECTORY WITH SINUSOIDAL INPUT

# Plotting parameters
"""
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
"""
# Plot the response of x1, x2 vs. time
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$x(t)$ (units)')
# Plot data
ax.plot(t_sol2, r_sol2, label='$x_1(t)$', color='C0')
ax.plot(t_sol2, dotr_sol2, label='$x_2(t)$', color='C1')
ax.plot(t_sol2, u, label='$u(t)$', color='C2')
ax.legend(loc='upper right')
#fig.tight_layout()
plt.show()


#%%
# DISCRETIZATION

# rewrite system with noise
A = np.array([[0,1],[0,0]])
B = np.array([[0],[1]])
L = np.array([[0],[1]])
C = np.array([1,0])
D = 0  
Q = 0.01  # acceleration noise variance
R = 0.001 # position noise variance
kf = Kalman_Filter(A,B,L,C,D,Q,R)

# Discretize system
T = dt 
A_d, B_d, Q_d, R_d = kf.discretize(T)

#%%
# GENERATING SENSOR MEASUREMENTS
include_noise = True # toggle to turn on and off noise to test the filter
include_input = False


w1 = np.sqrt(Q_d[0][0])*np.random.randn(len(t))
w2 = np.sqrt(Q_d[0][1])*np.random.randn(len(t))
w = np.block([[w1], [w2]])

Q_c = Q/dt
w_d = np.sqrt(Q_c)*np.random.randn(len(t))
v = np.sqrt(R_d)*np.random.randn(len(t))
if include_input:
    a_sol = a_sol2
    r_sol = r_sol2
    if not include_noise:
        a_n = a_sol2 
        y_n = r_sol2 
    else:
        a_n = a_sol2 + w_d
        y_n = r_sol2 + v 
else:
    if not include_noise:
        a_n = a_sol
        y_n = r_sol 
    else:
        a_n = a_sol + w_d
        y_n = r_sol + v 


#%%
# PLOTTING TRAJECTORY WITH NOISE

fig, ax = plt.subplots(2, 1)
# Format axes
"""
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
"""
ax[0].set_xlabel(r'$t$ (s)')
ax[1].set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$x(t)$ (units)')
ax[1].set_ylabel(r'$a(t)$ (units/$s^2$)')

# Plot data
ax[0].plot(t, y_n, label='noisy position measurement', color='C1')
ax[0].plot(t, r_sol, label='true position', color='C0')
ax[1].plot(t, a_n, label='noisy acceleration measurement', color='C1')
ax[1].plot(t, a_sol, label='true acceleration', color='C0')
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
#fig.tight_layout()
plt.show()

#%%
# KALMAN FILTER
"""
- Like an observer but uses error covariances to optimally estimate states
- process noise: w_d disturbance with covariance v_d (nxn matrix)
- meas. noise: v noise with covariance v_n (nxn matrix)

"""
# Initial guesses
x_hat0 = np.array([[5], [0]]) # true value
P_hat0 = np.eye(2,2)
#x_hat0 = np.array([[4], [1]]) # somewhat close to true value
#P_hat0 = np.array([[0.9,0], [0,0.9]])
steps = int((t_end-t_start)/dt) # 10000 steps in 10 s range with dt = 1e-3
x_hat_k, P_hat_k, sigma3_k = kf.filter(a_n, y_n,x_hat0, P_hat0, steps, 1, 1)

# Extract out individual states
x1_hat = np.array(x_hat_k)[:, 0]
x2_hat = np.array(x_hat_k)[:, 1]
sigma3_1 = np.array(sigma3_k)[:,0]
sigma3_2 = np.array(sigma3_k)[:,1]

#%%
# PLOT ESTIMATED AND TRUE STATES
fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$x_1(t)$ (units)')
ax[1].set_ylabel(r'$x_2(t)$ (units/s)')
"""
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
"""
# Plot data
t_steps = np.arange(steps) * dt
#ax[0].plot(t, y_n, label='noisy position measurement', color='C1')
ax[0].plot(t, r_sol, label='true position', color='red')
ax[0].plot(t_steps, x1_hat, label='estimated position', color='C0')

ax[1].plot(t, dotr_sol, label='true velocity', color='red')
ax[1].plot(t_steps, x2_hat, label='estimated velocity', color='C0')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
#fig.tight_layout()
plt.show()

# %%
# PLOT ERROR
error_x1 = r_sol - x1_hat
error_x2 = dotr_sol - x2_hat

fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$e_{x_1}(t)$ (units)')
ax[1].set_ylabel(r'$e_{x_2}(t)$ (units/s)')
"""
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc("text", usetex=True)
"""
# Plot data
ax[0].plot(t_steps, error_x1, label='estimated position', color='C0')
#ax[0].plot(t_steps, sigma3_1, label = r'$3\sigma_1$', color = 'C1')
#ax[0].plot(t_steps, -sigma3_1, color = 'C1')
ax[0].fill_between(t_steps, sigma3_1, -sigma3_1, label = r'$3\sigma_1$', color = 'lightblue')
ax[1].plot(t_steps, error_x2, label='estimated velocity', color='C0')
#ax[1].plot(t_steps, sigma3_2, label = r'$3\sigma_2$', color = 'C1')
#ax[1].plot(t_steps, -sigma3_2, color = 'C1')
ax[1].fill_between(t_steps, sigma3_2, -sigma3_2, label = r'$3\sigma_2$', color = 'lightblue')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
plt.show()

# %%
# NEES TEST
true_states = np.column_stack((r_sol, dotr_sol))
steps = int((t_end-t_start)/dt) # 10000 steps in 10 s range with dt = 1e-3

d_2 = np.zeros(len(true_states))
random.seed(random.randint(1,10))
N = 100
for i in range(N):
    random.seed(random.randint(1,10))
    x_hat0 = np.array([[5 + random.uniform(-1,1)], [0 + random.uniform(-1,1)]])
    P_hat0 = np.eye(2,2) + np.array([[1 - random.random()/10, 0], [0, 1 - random.random()/10]])
    x_hat_k, P_hat_k, sigma3_k = kf.filter(a_n, y_n,x_hat0, P_hat0, steps, 1, 1)
    d_2_i = kf.nees(x_hat_k, true_states, P_hat_k)
    d_2 = d_2 + d_2_i

d_2 = d_2/(N)
alpha = 1 - 0.95
lower_confidence = alpha/2
upper_confidence = 1 - alpha/2
lower_bound = stats.chi2.ppf(lower_confidence, df=2)
upper_bound = stats.chi2.ppf(upper_confidence, df=2)
#d_2 = kf.nees(x_hat_k, true_states, P_hat_k)
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'${d_M}^2$')
ax.plot(t, d_2, label = r'$\bar{{d_M}^2}$')
plt.axhline(2, color = 'r', linestyle = '--', label = r'$E[{{d_M}^2}], k=2$')
plt.axhline(lower_bound, color = 'gray', linestyle = '--', label = r'${{d_M}^2}^{*}, k=2$')
plt.axhline(upper_bound, color = 'gray', linestyle = '--')
ax.legend(loc='upper right')
plt.show()

#stats.chi2.ppf(0.95)
"""
fig, ax = plt.subplots()
ax.plot(t, d_2)
plt.show()
"""

"""
- get a new d at each time step
- should converge to 2 with one trajectory bc 2 dof
- converge to 2 * N, num trials with monte carlo (divide, so averages out to 1)
- random covariance and IC centered around mean
- bounds are 2.5 and 97.5% --> 95

- 100 trials
- Nees over confident above condience bound, underconfident below lower boumd

"""

# %%

# Implement Kalman with position measurements at 1/10th the frequency of accelerometer measurements and repeat plots

# Can predict at every step but can only correct every 10 steps

# Initial guesses
x_hat_k2, P_hat_k2, sigma3_k2 = kf.filter(a_n, y_n,x_hat0, P_hat0, steps, 10, 1)

# Extract out individual states
x1_hat2 = np.array(x_hat_k2)[:, 0]
x2_hat2 = np.array(x_hat_k2)[:, 1]
sigma3_1_2 = np.array(sigma3_k2)[:,0]
sigma3_2_2 = np.array(sigma3_k2)[:,1]

#%%
# PLOT ESTIMATED AND TRUE STATES WITH DIFFERENT SAMPLING FREQUENCY
fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$x_1(t)$ (units)')
ax[1].set_ylabel(r'$x_2(t)$ (units/s)')
"""
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
"""
# Plot data
t_steps = np.arange(steps) * dt
#ax[0].plot(t, y_n, label='noisy position measurement', color='C1')
ax[0].plot(t, r_sol, label='true position', color='red')
ax[0].plot(t_steps, x1_hat2, label='estimated position', color='C0')

ax[1].plot(t, dotr_sol, label='true velocity', color='red')
ax[1].plot(t_steps, x2_hat2, label='estimated velocity', color='C0')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
#fig.tight_layout()
plt.show()

# %%
# PLOT ERROR WITH DIFFERENT SAMPLING FREQUENCY
error_x1_2 = r_sol - x1_hat2
error_x2_2 = dotr_sol - x2_hat2

fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$e_x1(t)$ (units)')
ax[1].set_ylabel(r'$e_x2(t)$ (units/s)')
"""
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
"""
# Plot data
ax[0].plot(t_steps, error_x1_2, label='estimated position', color='C0')
ax[0].fill_between(t_steps, sigma3_1_2, -sigma3_1_2, label = r'$3\sigma_1$', color = 'lightblue')
ax[1].plot(t_steps, error_x2_2, label='estimated velocity', color='C0')
ax[1].fill_between(t_steps, sigma3_2_2, -sigma3_2_2, label = r'$3\sigma_2$', color = 'lightblue')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
plt.show()

# %%
# NEES TEST WITH DIFFERENT SAMPLING FREQUENCY

