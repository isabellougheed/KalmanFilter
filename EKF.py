# Import packages
import numpy as np
import control
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import linalg
from scipy import stats
from MSD import MSD_NL_System
import random

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
        self.e_tol = 10 ** (-3)  # IS THIS GOOD??
        self.test = 0

    def discretize(self, T):
        # building large matrix
        xi = np.block(
            [
                [A, L @ (Q * (L.T)), np.zeros((2, 2)), np.zeros((2, 1))],
                [np.zeros((2, 2)), -A.T, np.zeros((2, 2)), np.zeros((2, 1))],
                [np.zeros((2, 2)), np.zeros((2, 2)), A, B],
                [
                    np.zeros((1, 2)),
                    np.zeros((1, 2)),
                    np.zeros((1, 2)),
                    np.zeros((1, 1)),
                ],
            ]
        )
        upsilon = linalg.expm(xi * T)

        # extract A_d, B_d, Q_d
        upsilon_11 = upsilon[0:2, 0:2]
        upsilon_12 = upsilon[0:2, 2:4]
        upsilon_34 = upsilon[4:6, 6]
        A_d = upsilon_11
        B_d = np.array(upsilon_34)  # it transposes for some reason
        B_d = np.array([[upsilon_34[0]], [upsilon_34[1]]])
        Q_d = upsilon_12 @ (upsilon_11.T)
        R_d = R / T

        # initialize parameters
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
        x_1 = x[0][0]
        d = msd.d
        h = msd.h
        C_k = np.array([(d + x_1) / (np.sqrt((d + x_1) ** 2 + h**2)), 0])
        # just to double check my math for the jacobian
        dg_dx = (
            np.sqrt(((d + x_1 + 0.1) ** 2 + h**2))
            - np.sqrt(((d + x_1 - 0.1) ** 2 + h**2))
        ) / 0.2
        if np.abs(dg_dx - C_k[0]) > 0.01:
            print("flag")
        return C_k

    def predict(self, x_prior, P_prior, u):
        xk_p = A_d @ x_prior + B_d * u
        Pk_p = A_d @ P_prior @ (A_d.T) + Q_d
        return xk_p, Pk_p

    def correct_ekf(self, msd, xk_p, Pk_p, y):
        C_k = self.c_k(msd, xk_p)
        S_k = C_k @ Pk_p @ (C_k.T) + R_d
        K_k = Pk_p @ (C_k.T) * ((S_k) ** (-1))

        # just reshaping because I think I am flattening some of my variables sometimes
        K_k = np.array([[K_k[0]], [K_k[1]]])
        C_k = np.array([[C_k[0], 0]])

        x_k = xk_p + K_k * (y - msd.g(xk_p))

        # P_k = (np.eye(2,2) - K_k @ C_k) @ Pk_p @(np.eye(2,2) - K_k @ C_k).T + K_k*R_d*K_k.T
        ### TRY BOTH

        P_k = (np.eye(2, 2) - K_k @ C_k) @ Pk_p
        P_k = (P_k + P_k.T) / 2  # forcing symmetry

        return K_k, x_k, P_k

    def correct_iekf(self, msd, xk_p, Pk_p, y):
        counter = 1
        # first iteration
        x_k_j = xk_p
        C_k = self.c_k(msd, xk_p)
        K_k = Pk_p @ (C_k.T) * (C_k @ Pk_p @ (C_k.T) + R_d) ** (-1)

        # just reshaping because I think I am flattening some of my variables sometimes
        K_k = np.array([[K_k[0]], [K_k[1]]])
        C_k = np.array([[C_k[0], 0]])

        x_k_j_1 = xk_p + K_k * (y - msd.g(xk_p))

        P_k = (np.eye(2, 2) - K_k @ C_k) @ Pk_p
        # P_k = (np.eye(2,2) - K_k @ C_k) @ Pk_p @(np.eye(2,2) - K_k @ C_k).T + K_k*R_d*K_k.T
        P_k_j_1 = 0.5 * (P_k + P_k.T)  # forcing symmetry

        while np.linalg.norm(x_k_j - x_k_j_1) > 0.001 and counter < 10:
            x_k_j = x_k_j_1
            P_k_j = P_k_j_1
            C_k = self.c_k(msd, x_k_j)
            K_k = Pk_p @ (C_k.T) * (C_k @ P_k_j @ (C_k.T) + R_d) ** (-1)

            # just reshaping because I think I am flattening some of my variables sometimes
            K_k = np.array([[K_k[0]], [K_k[1]]])
            C_k = np.array([[C_k[0], 0]])

            x_k_j_1 = xk_p + K_k * (y - msd.g(x_k_j))

            P_k = (np.eye(2, 2) - K_k @ C_k) @ Pk_p
            # P_k = (np.eye(2,2) - K_k @ C_k) @ P_k_j @(np.eye(2,2) - K_k @ C_k).T + K_k*R_d*K_k.T
            P_k_j_1 = 0.5 * (P_k + P_k.T)  # forcing symmetry

            counter += 1

        return K_k, x_k_j_1, P_k_j_1, counter

    def filter(self, msd, a_n, y_n, x_hat0, P_hat0, steps, frequ_a, frequ_y, isIEKF):
        # Making a list to add all estimated states and covariances at each step into
        x_hat_k = []
        P_hat_k = []
        sigma3_k = []
        iterations = []

        # initializing lists with initial guesses
        x_hat_k.append(x_hat0.flatten())
        P_hat_k.append(P_hat0)
        sigma3_k.append(3 * np.sqrt(np.diag(P_hat0)))

        # Setting up x_k-1, P_k-1
        x_k_1 = x_hat0
        P_k_1 = P_hat0

        for i in range(1, steps):
            # predict
            xk_p, Pk_p = self.predict(x_k_1, P_k_1, a_n[i - 1])

            # just to get K_k
            if isIEKF:
                K_k, x__, P__, counter = self.correct_iekf(msd, xk_p, Pk_p, y_n[i])
            else:
                K_k, x__, P__ = self.correct_ekf(msd, xk_p, Pk_p, y_n[i])

            if frequ_a != frequ_y:
                # if sampling frequencies not the same, can't do correction every step because missing y measurements
                if i % (frequ_a / frequ_y) == 0:
                    # if this step has a y measurement, can do correction
                    if isIEKF:
                        K_k, x_k, P_k, counter = self.correct_iekf(
                            msd, xk_p, Pk_p, y_n[i]
                        )
                        iterations.append(counter)
                    else:
                        K_k, x_k, P_k = self.correct_ekf(msd, xk_p, Pk_p, y_n[i])
                        iterations.append(1)

                else:
                    # skip correction
                    x_k = xk_p
                    P_k = Pk_p
            else:
                # can do correction every time if a,y have same sampling rate
                if isIEKF:
                    K_k, x_k, P_k, counter = self.correct_iekf(msd, xk_p, Pk_p, y_n[i])
                    iterations.append(counter)
                else:
                    K_k, x_k, P_k = self.correct_ekf(msd, xk_p, Pk_p, y_n[i])
                    iterations.append(1)

            # next iteration
            x_k_1 = x_k
            P_k_1 = P_k

            # add to arrays
            x_hat_k.append(x_k.flatten())
            P_hat_k.append(P_k)
            S = np.diag(P_k)
            sigma3_k.append(3 * np.sqrt(S))

        return x_hat_k, P_hat_k, sigma3_k, iterations

    def nees(self, estimated_states, true_states, P):
        """
        This method takes an array of all estimated states at all time steps, an array of all covariance at all time steps, and an array of all true states at all time steps.

        This method returns the mahalanobis distances at each iteration in an array.
        """

        d_2 = np.zeros(len(estimated_states))
        for i in range(len(estimated_states)):
            e = estimated_states[i] - true_states[i]
            d_2[i] = (e.T) @ (np.linalg.inv(P[i])) @ e

        return d_2


# %%
# set up system
m = 2.5
k = 30
c = 3
A = 1
w = 5
# h = 0.5
h = 1
d = 6
sys = MSD_NL_System(m, c, k, A, w, h, d)
x0 = np.array([[5], [0]])

# Time
dt = 1e-3
t_start = 0
t_end = 10
t = np.arange(t_start, t_end, dt)  # t has 10000 steps

# %%

# GENERATING GROUND TRUTH FOR NO INPUT
sol = integrate.solve_ivp(
    sys.f1,
    (t_start, t_end),
    x0.ravel(),
    t_eval=t,
    rtol=1e-6,
    atol=1e-6,
    method="RK45",
)  # I like this better because it returns all of the solution steps and you don't need to iterate through
t_sol = sol.t
x_sol = sol.y
r_sol = x_sol[0]
dotr_sol = x_sol[1]
a_sol = (1 / m) * (-k * r_sol - c * dotr_sol)
y_sol = np.sqrt((r_sol + d) ** 2 + h**2)

# %%

# GENERATING GROUND TRUTH FOR SINUSOIDAL INPUT
sol2 = integrate.solve_ivp(
    sys.f2,
    (t_start, t_end),
    x0.ravel(),
    t_eval=t,
    rtol=1e-6,
    atol=1e-6,
    method="RK45",
)  # I like this better because it returns all of the solution steps and you don't need to iterate through
t_sol2 = sol2.t
x_sol2 = sol2.y
r_sol2 = x_sol2[0]
dotr_sol2 = x_sol2[1]
u = sys.input(t_sol2)
a_sol2 = (1 / m) * (u - k * r_sol2 - c * dotr_sol2)
y_sol2 = np.sqrt((r_sol2 + d) ** 2 + h**2)

# %%

# DISCRETIZATION

# rewrite system with noise
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])
L = np.array([[0], [1]])
C = np.array([d / (np.sqrt(d**2 + h**2)), 0])
D = 0
Q = 0.001  # acceleration noise variance
# Q = 0.01
R = 0.00001  # position noise variance
R = 0.00001
# R = 0.0001
Q = 0.1
R = 0.01
ekf = EKF(A, B, L, C, D, Q, R)

# Discretize system
T = dt
A_d, B_d, Q_d, R_d = ekf.discretize(T)

# %%

# GENERATING SENSOR MEASUREMENTS
isIEKF = True
include_noise = True  # toggle to turn on and off noise to test the filter
include_input = False
Q_c = Q / dt
w_d = np.sqrt(Q_c) * np.random.randn(len(t))
v = np.sqrt(R_d) * np.random.randn(len(t))
if include_input:
    a_sol = a_sol2
    r_sol = r_sol2
    y_sol = y_sol2
    if not include_noise:
        a_n = a_sol2  # change input depending on if zero input or sinusoidal
        y_n = np.sqrt((r_sol2 + d) ** 2 + h**2)
    else:
        a_n = a_sol2 + w_d
        y_n = np.sqrt((r_sol2 + d) ** 2 + h**2) + v
else:
    if not include_noise:
        w = np.zeros((len(t), 1))
        v = np.zeros((len(t), 1))
        a_n = a_sol  # change input depending on if zero input or sinusoidal
        y_n = np.sqrt((r_sol + d) ** 2 + h**2)
    else:
        a_n = a_sol + w_d
        y_n = np.sqrt((r_sol + d) ** 2 + h**2) + v

# %%

# PLOTTING TRAJECTORY WITH NOISE
plt.rc("figure", figsize=(11.5, 8.5))
plt.rc("font", family="Times New Roman", size=20)
plt.rc("axes", grid=True, labelsize=20)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("grid", linestyle="--")
plt.rcParams["lines.markersize"] = 2

fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r"$t$ (s)")
ax[0].set_ylabel(r"$y(t)$ (m)")
ax[1].set_ylabel(r"$a(t)$ (m/$s^2$)")


# Plot data
ax[0].plot(t, y_n, label="noisy y measurement", color="C1")
ax[0].plot(t, y_sol, label="true y", color="C0")
ax[1].plot(t, a_n, label="noisy acceleration measurement", color="C1")
ax[1].plot(t, a_sol, label="true acceleration", color="C0")
ax[0].legend(loc="upper right")
ax[1].legend(loc="upper right")
fig.tight_layout()
plt.show()

# %%

# EXTENDED KALMAN FILTER
# Initial guesses
x_hat0 = np.array([[5], [0]])  # true value
P_hat0 = np.eye(2, 2)
# x_hat0 = np.array([[4], [1]]) # somewhat close to true value
# P_hat0 = np.array([[0.9,0], [0,0.9]])
steps = int((t_end - t_start) / dt)  # 10000 steps in 10 s range with dt = 1e-3
x_hat_k, P_hat_k, sigma3_k, iterations = ekf.filter(
    sys, a_n, y_n, x_hat0, P_hat0, steps, 1, 1, isIEKF
)

# Extract out individual states
x1_hat = np.array(x_hat_k)[:, 0]
x2_hat = np.array(x_hat_k)[:, 1]
sigma3_1 = np.array(sigma3_k)[:, 0]
sigma3_2 = np.array(sigma3_k)[:, 1]

# %%

# PLOT ESTIMATED AND TRUE STATES

fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r"$t$ (s)")
ax[0].set_ylabel(r"$x_1(t)$ (m)")
ax[1].set_ylabel(r"$x_2(t)$ (m/s)")

# Plot data
t_steps = np.arange(steps) * dt
# ax[0].plot(t, y_n, label='noisy position measurement', color='C1')
ax[0].plot(t_steps, x1_hat, label="estimated position", color="C0")
ax[0].plot(t, r_sol, label="true position", color="red")

ax[1].plot(t_steps, x2_hat, label="estimated velocity", color="C0")
ax[1].plot(t, dotr_sol, label="true velocity", color="red")

ax[0].legend(loc="upper right")
ax[1].legend(loc="upper right")
fig.tight_layout()
plt.show()

# %%

# PLOT ERROR
error_x1 = r_sol - x1_hat
error_x2 = dotr_sol - x2_hat

fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r"$t$ (s)")
ax[0].set_ylabel(r"$e_{x_1}(t)$ (m)")
ax[1].set_ylabel(r"$e_{x_2}(t)$ (m/s)")

# Plot data
ax[0].plot(t_steps, error_x1, label="estimated position", color="C0")
ax[0].fill_between(
    t_steps, sigma3_1, -sigma3_1, label=r"$3\sigma_1$", color="lightblue"
)
ax[1].plot(t_steps, error_x2, label="estimated velocity", color="C0")
ax[1].fill_between(
    t_steps, sigma3_2, -sigma3_2, label=r"$3\sigma_2$", color="lightblue"
)

ax[0].legend(loc="upper right")
ax[1].legend(loc="upper right")
ax[1].set_ylim(-3, 3)
ax[0].set_ylim(-3, 3)
fig.tight_layout()
plt.show()


# %%
# PLOT ITERATIONS IEKF
dt = 1e-3
t_start = 0
t_end = 10 - 1e-3
t = np.arange(t_start, t_end, dt)
if isIEKF == True:
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"Iterations")
    ax.plot(t, iterations, "bo")
    plt.show()
# %%
# NEES TEST

true_states = np.column_stack((r_sol, dotr_sol))
steps = int((t_end - t_start) / dt)  # 10000 steps in 10 s range with dt = 1e-3

d_2 = np.zeros(len(true_states) - 1)
random.seed(random.randint(1, 10))
N = 10
sigma = 0.1
for i in range(N):
    random.seed(random.randint(1, 10))
    pertebation = sigma * np.random.randn(2)
    x_hat0 = np.array([[5 + pertebation[0]], [0 + pertebation[1]]])
    P_hat0 = (sigma**2) * np.eye(2, 2)

    # also generate new measurements
    w_d = np.sqrt(Q_c) * np.random.randn(len(t) + 1)
    v = np.sqrt(R_d) * np.random.randn(len(t) + 1)
    if include_input:
        a_sol = a_sol2
        r_sol = r_sol2
        y_sol = y_sol2
        if not include_noise:
            a_n = a_sol2  # change input depending on if zero input or sinusoidal
            y_n = np.sqrt((r_sol2 + d) ** 2 + h**2)
        else:
            a_n = a_sol2 + w_d
            y_n = np.sqrt((r_sol2 + d) ** 2 + h**2) + v
    else:
        if not include_noise:
            w = np.zeros((len(t), 1))
            v = np.zeros((len(t), 1))
            a_n = a_sol  # change input depending on if zero input or sinusoidal
            y_n = np.sqrt((r_sol + d) ** 2 + h**2)
        else:
            a_n = a_sol + w_d
            y_n = np.sqrt((r_sol + d) ** 2 + h**2) + v

    x_hat_k, P_hat_k, sigma3_k, iterations = ekf.filter(
        sys, a_n, y_n, x_hat0, P_hat0, steps, 1, 1, isIEKF
    )
    d_2_i = ekf.nees(x_hat_k, true_states, P_hat_k)
    d_2 = d_2 + d_2_i

d_2 = d_2 / (2 * N)
alpha = 1 - 0.997
lower_confidence = alpha / 2
upper_confidence = 1 - alpha / 2
lower_bound = stats.chi2.ppf(lower_confidence, df=2 * N)
upper_bound = stats.chi2.ppf(upper_confidence, df=2 * N)
# d_2 = kf.nees(x_hat_k, true_states, P_hat_k)

fig, ax = plt.subplots()
ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"NEES")
# ax.set_ylim(-1,20)
ax.plot(t, d_2, label=r"NEES")
plt.axhline(1, color="r", linestyle="--", label=r"Expected NEES")
plt.axhline(lower_bound / (2 * N), color="gray", linestyle="--", label=r"$99.7\%$ CI")
plt.axhline(upper_bound / (2 * N), color="gray", linestyle="--")
ax.legend(loc="upper right")
plt.show()
