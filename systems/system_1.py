import numpy as np
class model:
    def __init__(self):

        self.A = None # state to state transition
        self.B = None # input to state transition
        self.Const = None

        # State and Input constraints
        self.Hx = None
        self.hx = None
        self.Hu = None
        self.hu = None

    def get_plant(self):
        d = 0.5
        m = 1 / 0.8
        delta_t = 0.2
        self.A = np.array([[1, delta_t],[-delta_t / m, 1 - d * delta_t / m]])
        self.B = np.array([[0],[1]])
        self.Const = np.array([[0],[0.1]])

        Hx_up = np.eye((2))
        Hx_low = -np.eye((2))
        self.Hx = np.vstack((Hx_up,Hx_low))

        hx_up = np.ones((2))
        hx_up[0] = 1.2
        hx_up[1] = 0.8
        hx_low = 0 *np.ones((2))
        self.hx = np.hstack((hx_up,hx_low))

        Hu_up = np.eye((1)) # one input per node
        Hu_low = -np.eye((1)) # one input per node
        self.Hu = np.vstack((Hu_up,Hu_low))

        hu_up = 0.5 * np.ones((1)) # one input per node
        hu_low = 0.5 * np.ones((1)) # one input per node 
        self.hu = np.hstack((hu_up,hu_low))
