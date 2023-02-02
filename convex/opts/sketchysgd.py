import numpy as np
import scipy.sparse.linalg as la
from .sketchy_funcs import *

class SketchySGD():
    def __init__(self,
                model,
                update_freq,
                rank,
                rho,
                bh):
        self.model = model
        self.update_freq = update_freq
        self.rank = rank
        self.bh = bh
        self.eta = None
        self.rho = rho
        self.n_iters = 0
        self.U = None
        self.S = None
        self.S_mod = None

    def step(self, indices):
        # Update the preconditioner at the appropriate frequency
        if self.n_iters % self.update_freq['precond'] == 0:
                h_indices = np.random.choice(self.model.ntr, self.bh, replace = False)
                self.U, self.S = self.model.get_rand_nys_appx(h_indices, self.rank)

                lin_op = get_lin_op(self.model.get_hessian_diag(h_indices), 
                    self.model.Atr[h_indices, :], 
                    self.U, 
                    self.S, 
                    self.rho)
                sing_val = la.svds(lin_op, k = 1, which = 'LM', return_singular_vectors = False)
                # print(f'Largest eigenvalue = {sing_val[0] ** 2}')

                self.eta = sing_val[0] ** -2

                self.S_mod = np.divide(self.S, self.S + self.rho)

        
        g = self.model.get_grad(indices)
        if self.rank == 1:
            vns = appx_newton_step2(self.U, self.S_mod, self.rho, g)
        else:
            vns = appx_newton_step(self.U, self.S, self.rho, g)
        self.model.w -= self.eta * vns
        self.n_iters += 1
            