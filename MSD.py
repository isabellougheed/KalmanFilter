# Import packages
import numpy as np
import control
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import linalg

"""

This class is only to set up the system.

You don't need to run this file, it is just used by Kalman_Filter.py

"""

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
    
    
