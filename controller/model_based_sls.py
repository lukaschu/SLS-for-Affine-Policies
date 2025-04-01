from scipy.linalg import block_diag
from scipy.linalg import kron
import numpy as np
import cvxpy as cp

class model_based_solver_sls():
    def __init__(self, sys, params):
        # Needed for system repsonse constraints
        self.ZOB, self.RHS = self.get_zob_constraints(sys,params)

    def get_zob_constraints(self,sys,params):
        
        # Identity matrix for Kronecker product
        I = kron(np.eye(2), np.eye(params.T + 1))

        # Constructing Z
        Z = kron(np.eye(params.T), np.eye(2))
        Z = np.vstack((np.zeros(((2),(params.T)*(2))), Z))
        Z = np.hstack((Z, np.zeros(((2) * (params.T + 1), (2)))))

        # Creating block diagonal matrices
        A_repmat = [sys.A] * (params.T+1)
        B_repmat = [sys.B] * (params.T+1)

        A_block = block_diag(*A_repmat)
        B_block = block_diag(*B_repmat)

        # Combining matrices
        ZOB = np.hstack((I - Z @ A_block, -Z @ B_block))
        ZOB = ZOB[:, :-1]  # This slices the last column (We don t need them as last input is not relevant for MPC formulation)

        const_repeated = np.vstack([sys.Const] * params.T)
        RHS = np.block([ [np.eye(2), np.zeros((2,1))] , [np.zeros((params.T * 2,2)) * 2, const_repeated] ])

        return ZOB, RHS

    def _init_problem(self, sys, params, x_ref):
        # Define varibale phix and phiu
        self.Phi_x = cp.Variable((2 * (params.T+1), 3))
        self.Phi_u = cp.Variable((params.T, 3))
        self.x_0 = cp.Parameter((2))

        # Define system dynamics constraint 
        constraints = [self.ZOB @ cp.vstack((self.Phi_x, self.Phi_u)) == self.RHS]

        # define objective using x_ss and u_ss + state and input constraints
        state_traj = self.Phi_x[:, 0:2] @ self.x_0 + self.Phi_x[:, 2]
        input_traj = self.Phi_u[:, 0:2] @ self.x_0 + self.Phi_u[:, 2]

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

        input_sequence = self.Phi_u[:,0:2] @ self.x_0 + self.Phi_u[:,2]

        # return the first input (solved)
        return input_sequence[:1].value
    

