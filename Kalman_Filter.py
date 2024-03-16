# Import packages
import numpy as np
import control
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import linalg

class MSD_System:
    def __init__(self, m, c, k, A, w):
        #constructor
        self.m = m
        self.c = c
        self.k = k
        self.A = A
        self.w = w

    def f1(self, t, x):
        """Method for integration of ODE.

        dot_x = f(x, 0) given x0, given no input f(t)
        """
        # Extract states.
        dot_r = x[1]
        ddot_r = (-self.c*x[1]-self.k*x[0])/self.m 
        dot_x = np.vstack((dot_r, ddot_r))
        return dot_x.ravel()  # flatten the array
        #return np.array([[dot_r], [ddot_r]])
    
    def f2(self, t, x):
        """Method for integration of ODE.

        dot_x = f(x, u) given x0, given a sinusoidal input
        """
        # Extract states.
        dot_r = x[1]
        ddot_r = ((-self.c*x[1]-self.k*x[0])/self.m) + self.A*np.sin(self.w*t)
        dot_x = np.vstack((dot_r, ddot_r))
        return dot_x.ravel()  # flatten the array
        #return np.array([[dot_r], [ddot_r]])

    def input(self, t):
        return self.A*np.sin(self.w*t) 
    
    def state_space(self):
        # ground truth state space
        A = np.array([[0,1], [-self.k/self.m, -self.c/self.m]])
        B = np.array([[0],[1]])
        C = np.eye(2,2)
        D = np.zeros((2,1))
        return control.StateSpace(A,B,C,D)

# %%
# GENERATING GROUND TRUTH DATA FOR NO INPUT
           
# Time
dt = 1e-3
t_start = 0
t_end = 10
t = np.arange(t_start, t_end, dt) # t has 1000 steps

#set up system
m = 1
k = 10
c = 3
A = 1
w = 5
sys = MSD_System(m,c,k,A,w)
x0 = np.array([[5], [0]])

#sol = integrate.RK45(fun = sys.f, t0 = t_start, y0 = x0.ravel(), t_bound = t_end)
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
include_noise = False # toggle to turn on and off noise to test the filter
R = 0.15 # position noise variance
Q = 0.15 # acceleration noise variance
w = np.sqrt(Q)*np.random.randn(len(t))
v = np.sqrt(R)*np.random.randn(len(t))
"""
For random samples from N(\mu, \sigma^2), use:
sigma * np.random.randn(...) + mu
"""
#noisy sensor measurements
a_sol = (1/m)*(u - k*r_sol - c*dotr_sol)
a_n = (1/m)*(u - k*r_sol - c*dotr_sol) + w #change input depending on if zero input or sinusoidal
y_n = r_sol + v 

#%%
# PLOTTING TRAJECTORY WITH SINUSOIDAL INPUT

# Plotting parameters

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

# New state space system

# Zero order hold
"""
- get CT TF from s.s model (G)

T = 0.1 # assoc. with accelerometer meas. frequency
        # 1/T > 2*w_max,accelerometer
Gd = G.sample(T, method = 'zoh') # DT TF
Gd_ss = control.ss(Gd) # DT s.s. model

- Van Loan's method for Q_k-1

"""
# rewrite system with noise
"""
DOUBLE CHECK MY REWRITTEN SYSTEM
"""
A = np.array([[0,1],[0,0]])
B = np.array([[0],[1]])
L = np.array([[0],[1]])
B_a_w = np.array([[0,0], [1,1]]) # just so I can discretize with zoh tools, I will separate after
                                # IS THIS OKAY ??????????????
C = np.array([1,0])
D = 0
sys = control.StateSpace(A,B,C,D)

# Discretize system

T = 0.1 #CHECK IF THIS IS OKAY
sys_n = sys.sample(T,method = 'zoh')
print("sys_n is ", sys_n)
xi = np.block([[A, L@(Q*(L.T)), np.zeros((2,2)), np.zeros((2,1))], [np.zeros((2,2)), -A.T, np.zeros((2,2)), np.zeros((2,1))], [np.zeros((2,2)), np.zeros((2,2)), A, B], [np.zeros((1,2)),np.zeros((1,2)),np.zeros((1,2)),np.zeros((1,1))]])

upsilon = linalg.expm(xi*T)
print("upsilon is ", upsilon)

upsilon_11 = upsilon[0:2, 0:2]
upsilon_12 = upsilon[0:2, 2:4]
upsilon_34 = upsilon[4:6, 6]
A_d = upsilon_11
B_d = np.array(upsilon_34) #why is it transposed?
B_d = np.array([[upsilon_34[0]], [upsilon_34[1]]])
Q_d = upsilon_12@upsilon_11.T

#%%
# KALMAN FILTER
"""
- Like an observer but uses error covariances to optimally estimate states
- process noise: w_d Gaussian disturbance with covariance v_d (nxn matrix)
- meas. noise: v Gaussian noise with covariance v_n (nxn matrix)

L,P,E = control.dlqe(A,L,C,QN,RN) #linear quadratic estimator for discrete time systems
K_f = L #Kalman estimator gain
# P is soln to Riccati eqn, E is eig(A-LC)

d/dt[x, x-hatx].T = [[(A-B K_r), B K_r],[0, (A - K_f C)]] [x, x-hatx].T + [w_d, w_n].T

- maybe create_estimator_iosystem could be useful because this is what dlqe is doing behind
the scenes and uses a P estimate and returns xhat

"""
x_hat0 = np.array([[5], [0]])
P_hat0 = np.eye(2,2)

#THIS ISNT BEST WAY TO DO THIS !!!!!!!!!!!!!
# ARRAYS IN ARRAYS
# Making a list to add all estimated states and covariances at each step into
x_hat_k = []
x_hat_1 = []
x_hat_2 = []
P_hat_k = []

x_hat_k.append(x_hat0.flatten())
P_hat_k.append(P_hat0)
x_hat_1.append(x_hat0[0,0])
x_hat_2.append(x_hat0[1,0])

x_k_1 = x_hat0
P_k_1 = P_hat0
# SHOULD THESE BE THE MEASURED a_n AND y_n ?????
a_k_1 = (1/m)*(u - k*x_hat0[0] - c*x_hat0[1]) + w[0]
y_k = C@x_hat0 + v[0] 

steps = 10000
for i in range(0, steps-1): #fix range
    # prediction
    xk_p = A_d @ x_k_1 + B_d*a_n[i] #x_k+1 = (A_k)(x_k) + (B_k)(u_k)
    Pk_p = A_d @ P_k_1 @ A_d.T + Q_d  #P_k+1 = (A_k)(P_k)(A_k).T + Q_k
    # correction
    K_k = Pk_p @ (C.reshape(1, -1)).T *(C @ Pk_p @ (C.reshape(1, -1)).T + R)**(-1)  ##### CHECK THIS R
    x_k = xk_p + K_k *(y_n[i] - C @ xk_p) #x_k+1|k+1 = x_k+1|k + K_k+1(y_k+1 - (C_k+1)(x_k+1|k))
    P_k = (np.eye(2,2) - K_k @ C.reshape(1, -1)) @ Pk_p @(np.eye(2,2) - K_k @ C.reshape(1, -1)).T + K_k*R*K_k.T # P_k+1|k+1 = (I - (K_k+1)(C_k+1))P_k+1|k(I - (K_k+1)(C_k+1)).T + (K_k+1)(R_k+1)(K_k+1).T
  ############ CHECK THIS R
   
    # next iteration
    x_k_1 = x_k
    P_k_1 = P_k

    x_hat_k.append(x_k.flatten())
    P_hat_k.append(P_k)

    if i == 0:
        print("Predicted")
        print("xk_p: ",xk_p)
        print("Pk_p: ",Pk_p)
        print("Estimated")
        print("x_k: ",x_k)
        print("P_k: ",P_k)

x1_hat = np.array(x_hat_k)[:, 0]
x2_hat = np.array(x_hat_k)[:, 1]

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
ax[0].plot(t, r_sol, label='true position', color='C0')
ax[0].plot(t_steps, x1_hat, label='estimated position', color='C2')

ax[1].plot(t, dotr_sol, label='true velocity', color='C0')
ax[1].plot(t_steps, x2_hat, label='estimated velocity', color='C2')

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
ax[0].plot(t_steps, error_x1, label='estimated position', color='C2')
ax[1].plot(t_steps, error_x2, label='estimated velocity', color='C2')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
fig.tight_layout()
plt.show()

# %%
