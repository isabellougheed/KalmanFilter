# Import packages
import numpy as np
import control
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import linalg
from MSD import MSD_System

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
        Q_d = upsilon_12@upsilon_11.T
        R_d = R/T

        #initialize parameters
        self.A_d = A_d
        self.B_d = B_d
        self.Q_d = Q_d
        self.R_d = R_d
        return A_d, B_d, Q_d, R_d
    
    def filter(self, a_n, y_n, x_hat0, P_hat0, steps, frequ_a, frequ_y):
        # Making a list to add all estimated states and covariances at each step into
        x_hat_k = []
        P_hat_k = []
        sigma3_k = []

        # initializing lists with initial guesses
        x_hat_k.append(x_hat0.flatten())
        P_hat_k.append(P_hat0)
        U0, S0, V0 = np.linalg.svd(P_hat0)
        sigma3_k.append(S0)

        #Setting up x_k-1, P_k-1
        x_k_1 = x_hat0
        P_k_1 = P_hat0

        for i in range(0, steps-1):
            # prediction
            xk_p = A_d @ x_k_1 + B_d*a_n[i] #x_k+1 = (A_k)(x_k) + (B_k)(u_k)
            Pk_p = A_d @ P_k_1 @ A_d.T + Q_d  #P_k+1 = (A_k)(P_k)(A_k).T + Q_k

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
            #U, S, V = np.linalg.svd(P_k) #changes order of singular values
            S = np.diag(P_k) #try this instead
            sigma3_k.append(3*np.sqrt(S))

        return x_hat_k, P_hat_k, sigma3_k


# %%
# GENERATING GROUND TRUTH DATA FOR NO INPUT
           
# Time
dt = 1e-3
t_start = 0
t_end = 10
t = np.arange(t_start, t_end, dt) # t has 10000 steps

#set up system
m = 2.5
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
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Plot the response of x1, x2 vs. time
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$x(t)$ (units)')
# Plot data
ax.plot(t_sol, r_sol, label='$x_1(t)$', color='C0')
ax.plot(t_sol, dotr_sol, label='$x_2(t)$', color='C1')
ax.legend(loc='upper right')
fig.tight_layout()
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
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Plot the response of x1, x2 vs. time
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$x(t)$ (units)')
# Plot data
ax.plot(t_sol2, r_sol2, label='$x_1(t)$', color='C0')
ax.plot(t_sol2, dotr_sol2, label='$x_2(t)$', color='C1')
ax.plot(t_sol2, u, label='$u(t)$', color='C2')
ax.legend(loc='upper right')
fig.tight_layout()
plt.show()

#%%
# GENERATING SENSOR MEASUREMENTS
include_noise = True # toggle to turn on and off noise to test the filter
include_input = False
R = 0.01 # position noise variance
Q = 0.01 # acceleration noise variance

w = np.sqrt(Q)*np.random.randn(len(t))
v = np.sqrt(R)*np.random.randn(len(t))
if include_input:
    if not include_noise:
        w = np.zeros((len(t),1))
        v = np.zeros((len(t),1))
        a_n = a_sol  #change input depending on if zero input or sinusoidal
        y_n = r_sol 
    else:
        a_n = a_sol + w 
        y_n = r_sol + v 
else:
    if not include_noise:
        w = np.zeros((len(t),1))
        v = np.zeros((len(t),1))
        a_n = a_sol2  #change input depending on if zero input or sinusoidal
        y_n = r_sol2 
    else:
        a_n = a_sol2 + w 
        y_n = r_sol2 + v 


#%%
# PLOTTING TRAJECTORY WITH SINUSOIDAL INPUT

fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$x(t)$ (units)')
ax[1].set_ylabel(r'$a(t)$ (units/s^2)')

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Plot data
ax[0].plot(t, y_n, label='noisy position measurement', color='C1')
ax[0].plot(t, r_sol, label='true position', color='C0')
ax[1].plot(t, a_n, label='noisy acceleration measurement', color='C1')
ax[1].plot(t, a_sol, label='true acceleration', color='C0')
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
fig.tight_layout()
plt.show()
#%%
# DISCRETIZATION

# rewrite system with noise
A = np.array([[0,1],[0,0]])
B = np.array([[0],[1]])
L = np.array([[0],[1]])
C = np.array([1,0])
D = 0
kf = Kalman_Filter(A,B,L,C,D,Q,R)

# Discretize system
T = dt 
A_d, B_d, Q_d, R_d = kf.discretize(T)

#%%
# KALMAN FILTER
"""
- Like an observer but uses error covariances to optimally estimate states
- process noise: w_d disturbance with covariance v_d (nxn matrix)
- meas. noise: v noise with covariance v_n (nxn matrix)

"""
# Initial guesses
#x_hat0 = np.array([[5], [0]]) # true value
#P_hat0 = np.eye(2,2)
x_hat0 = np.array([[4], [1]]) # somewhat close to true value
P_hat0 = np.array([[0.9,0], [0,0.9]])
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

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Plot data
t_steps = np.arange(steps) * dt
#ax[0].plot(t, y_n, label='noisy position measurement', color='C1')
ax[0].plot(t, r_sol, label='true position', color='red')
ax[0].plot(t_steps, x1_hat, label='estimated position', color='C0')

ax[1].plot(t, dotr_sol, label='true velocity', color='red')
ax[1].plot(t_steps, x2_hat, label='estimated velocity', color='C0')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
fig.tight_layout()
plt.show()

# %%
# PLOT ERROR
error_x1 = r_sol - x1_hat
error_x2 = dotr_sol - x2_hat

fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$e_x1(t)$ (units)')
ax[1].set_ylabel(r'$e_x2(t)$ (units/s)')

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Plot data
ax[0].plot(t_steps, error_x1, label='estimated position', color='C0')
ax[0].plot(t_steps, sigma3_1, label = r'$3\sigma_1$', color = 'C1')
ax[0].plot(t_steps, -sigma3_1, color = 'C1')
ax[1].plot(t_steps, error_x2, label='estimated velocity', color='C0')
ax[1].plot(t_steps, sigma3_2, label = r'$3\sigma_2$', color = 'C1')
ax[1].plot(t_steps, -sigma3_2, color = 'C1')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
plt.show()

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

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Plot data
t_steps = np.arange(steps) * dt
#ax[0].plot(t, y_n, label='noisy position measurement', color='C1')
ax[0].plot(t, r_sol, label='true position', color='red')
ax[0].plot(t_steps, x1_hat2, label='estimated position', color='C0')

ax[1].plot(t, dotr_sol, label='true velocity', color='red')
ax[1].plot(t_steps, x2_hat2, label='estimated velocity', color='C0')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
fig.tight_layout()
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

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Plot data
ax[0].plot(t_steps, error_x1_2, label='estimated position', color='C0')
ax[0].plot(t_steps, sigma3_1_2, label = r'$3\sigma_1$', color = 'C1')
ax[0].plot(t_steps, -sigma3_1_2, color = 'C1')
ax[1].plot(t_steps, error_x2_2, label='estimated velocity', color='C0')
ax[1].plot(t_steps, sigma3_2_2, label = r'$3\sigma_2$', color = 'C1')
ax[1].plot(t_steps, -sigma3_2_2, color = 'C1')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
plt.show()