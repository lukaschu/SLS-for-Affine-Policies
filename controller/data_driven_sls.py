from scipy.linalg import block_diag
from scipy.linalg import kron
from scipy.linalg import solve_discrete_are
import numpy as np
import cvxpy as cp

class data_driven_solver_sls():
    def __init__(self, sys, params):
        self.Hank_x = None
        self.Hank_u = None
        self.Hank_1 = None
    
    def _init_hankel(self,sys,params):
        Nu = 1
        Nx = 2
        L = params.T + 1 # over time horizon T and initial condition!!
        required_traj_length =  Nu * (Nx + L + 1) + Nx + L # u_traj needs to be PE of order n + (T+1) + 1 

        x_traj, u_traj = self.generate_traj(sys.A, sys.B, sys.Const, required_traj_length)

        # check if PE requirment is fullfiled
        PE_hankel = self.hankelize(u_traj, Nx + L + 1) # must have full rank!
        if(np.linalg.matrix_rank(PE_hankel) == Nu*(L+Nx+1)):
            self.Hank_x = self.hankelize(x_traj,L)
            self.Hank_u = self.hankelize(u_traj,L)[:-1,:] # We do not need the last row because we only optimize over T inputs
            self.Hank_1 = self.Hank_x[0:2,:] # the first n rows, here n = 2
        else:
            print("Input sequence is not PE of order L + Nx + 1: Try to construct a different trajectory")
            exit(1)

    def generate_traj(self, A, B, S, T):
        # Get parameters
        Nx, Nu = B.shape
        xtraj = np.zeros((Nx, T + 1))
        utraj = np.zeros((Nu, T))

        # Random initial condition
        xtraj[:, 0] = np.random.rand(2)

        # Solve the discrete algebraic Riccati equation for LQR controller
        P = solve_discrete_are(A, B, np.eye(Nx), np.eye(Nu))
        K = np.linalg.inv(B.T @ P @ B + np.eye(Nu)) @ (B.T @ P @ A)

        # Simulate trajectory using (LQR controller + noise)
        for t in range(T):
            x = xtraj[:, t]
            u_w = np.random.rand(Nu) * 0.03 * 2 - 0.03 
            u = -K @ x + u_w
            xtraj[:, t + 1] = A @ x + B @ u + S.reshape(-1)
            utraj[:, t] = u

        return xtraj[:,:-1], utraj
    
    def hankelize(self, signal, L):
        s, T = signal.shape  # s denotes the signal dimension
        H = np.zeros((L * s, T - L + 1))

        for i in range(L):
            H[i * s:(i + 1) * s, :] = signal[:, i:T - L + i + 1]

        return H
        
    def _init_problem(self, sys, params, x_ref):
        # Initialize the hankel matrrices
        self._init_hankel(sys, params)

        self.x_0 = cp.Parameter((2))
        size = self.Hank_1.shape[1], self.x_0.shape[0]
        self.G = cp.Variable((size[0],size[1]))
        self.g_hat = cp.Variable((size[0]))

        # declare the objective 
        objective = 0
        constraints = []

        constraints += [self.Hank_1 @ self.G == np.identity((size[1]))]
        constraints += [self.Hank_1 @ self.g_hat == 0]

        # If system affine (and not linear)
        constraints += [self.G.T @ np.ones((self.G.shape[0])) == np.ones((self.G.shape[1]))]
        constraints += [np.ones((1,self.g_hat.shape[0])) @ self.g_hat == 1]

        # This represents g_hat @ 1_{n}.T (a concatentation of vectors: g_hat_matrix = [g_hat, g_hat ..., g_hat])
        g_hat_matrix = cp.hstack([cp.reshape(self.g_hat,(-1,1))]*self.G.shape[1])

        # We can now compute the clsoed loop maps
        state_traj = self.Hank_x @ (self.G - g_hat_matrix) @ self.x_0 + self.Hank_x @ self.g_hat
        input_traj = self.Hank_u @ (self.G - g_hat_matrix) @ self.x_0 + self.Hank_u @ self.g_hat

        objective = 0.0

        for i in range(params.T):
            state = state_traj[2*i:2*(i+1)]
            input = input_traj[i:i+1]
            objective += cp.quad_form(state- x_ref,params.Q) 
            # objective += input.T @ params.R @ input

            constraints += [sys.Hx @ state <= sys.hx]
            constraints += [sys.Hu @ input <= sys.hu]

        last_state = state_traj[-2:]
        objective += cp.quad_form(last_state- x_ref,params.Q) 
        constraints += [sys.Hx @ last_state <= sys.hx]

        self.prob = cp.Problem(cp.Minimize(objective), constraints)
    
    def solve(self,x_init):
        self.x_0.value = x_init
        self.prob.solve(verbose=True, solver='ECOS')

        g_hat_matrix = cp.hstack([cp.reshape(self.g_hat,(-1,1))]*self.G.shape[1])
        input_sequence = self.Hank_u @ (self.G - g_hat_matrix) @ self.x_0 + self.Hank_u @ self.g_hat

        # return the first input (solved)
        return input_sequence[:1].value