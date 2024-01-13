import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from .pcg import PCG

class SparseSPPCG(PCG):
    def __init__(self, model, s):
        super().__init__(model, s)
   
    def _generate_embedding(self):
        n = self.model.ntr
        zeta = min(self.s, 8)
        # rows = np.random.choice(range(r), zeta * self.s) # Faster choice that allows for repeated indices
        rows = np.random.rand(n,self.s).argsort(axis = -1)
        rows = rows[:, :zeta].reshape(-1)
        cols = np.kron(range(n), np.ones(zeta))
        signs = np.sign(np.random.uniform(0, 1.0, len(cols)) - 0.5)
        Omega = csr_matrix((signs/np.sqrt(zeta), (rows, cols)), shape = (self.s, n))
        return Omega
   
    def _construct_precond(self):
        Omega = self._generate_embedding()
        Y = Omega@self.model.Atr / np.sqrt(self.model.ntr)

        if self.s >= self.model.Atr.shape[1]:
            G = Y.T@Y+self.model.mu*np.eye(self.model.Atr.shape[1])
            L = la.cholesky(G,lower = True)
            self.L = L
        else:
            G = Y@Y.T+self.model.mu*np.eye(self.s)
            self.L = la.cholesky(G,lower = True)
            self.Y = Y

    def _apply_precond(self, v):
        if self.s >= self.model.Atr.shape[1]:
            L_inv_v = la.solve_triangular(self.L, v, trans = 0, lower = True, check_finite = False)
            P_inv_v = la.solve_triangular(self.L, L_inv_v, trans = 1, lower = True, check_finite = False)
        else:
            Yv = self.Y@v
            L_inv_Yv = la.solve_triangular(self.L, Yv, trans = 0, lower = True, check_finite = False)
            LT_inv_L_inv_Yv = la.solve_triangular(self.L, L_inv_Yv, trans = 1, lower = True, check_finite = False)
            P_inv_v = 1/self.model.mu*(v-self.Y.T@LT_inv_L_inv_Yv)
        return P_inv_v