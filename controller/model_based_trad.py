from scipy.linalg import block_diag
from scipy.linalg import kron
import numpy as np
import cvxpy as cp

class model_based_solver_trad():

    def _init_problem(self, sys, params,x_ref):
        # Define varibale phix and phiu
        self.x = cp.Variable((2, params.T+1))
        self.u = cp.Variable((1, params.T))
        self.x_0 = cp.Parameter((2))

        objective = cp.quad_form(self.x[:,0] - x_ref,params.Q)
        constraints = [self.x[:,0] == self.x_0]
        constraints += [sys.Hx @ self.x[:,0] <= sys.hx]

        for i in range(params.T):
            constraints += [self.x[:,i+1] == sys.A @ self.x[:,i] + sys.B @ self.u[:,i] + sys.Const.reshape(-1)]

            objective += cp.quad_form(self.x[:,i+1]- x_ref,params.Q)
            # objective += cp.quad_form(self.u[:,i], params.R)

            constraints += [sys.Hx @ self.x[:,i+1] <= sys.hx]
            constraints += [sys.Hu @ self.u[:,i] <= sys.hu]

        self.prob = cp.Problem(cp.Minimize(objective), constraints)
    
    def solve(self,x_init):
        self.x_0.value = x_init
        self.prob.solve(verbose=True, solver='ECOS')

        # return the first input (solved)
        return self.u[0,0].value
    

