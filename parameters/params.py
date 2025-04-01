## a class which contains all relevant parameters for the optimization problem
import numpy as np

class parameters():
    def __init__(self):
        self.T = 10 # Time horizon for optimization
        self.t_horizon = 15 # simulation horizon

        self.Q = np.eye(2)
        self.Q[1,1] = 0
        self.R = np.eye(1)
        self.x_ref = 0.2
        self.u_ref = 0.3

        # Define initial condition
        self.x0 = []