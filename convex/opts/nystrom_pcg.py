import numpy as np
import scipy as sp
import scipy.linalg as la
from .pcg import PCG

class NystromPCG(PCG):
    def __init__(self, model, s):
        super().__init__(model, s)
    
    def _construct_precond(self):
        Omega = np.random.randn(self.model.Atr.shape[1], self.s)
        Omega = la.qr(Omega, mode='economic')[0]
        Y = 1/self.model.ntr*self.model.Atr.T@(self.model.Atr@Omega)
        v = np.sqrt(self.model.Atr.shape[1])*np.spacing(np.linalg.norm(Y,2))
        Yv = Y+v*Omega
        Core = Omega.T@Yv
        try:
            C = np.linalg.cholesky(Core)
        except:
            eig_vals = la.eigh(Core,eigvals_only=True)
            v = v+np.abs(np.min(eig_vals))
            Core = Core+v*np.eye(self.s)
            C = np.linalg.cholesky(Core)

        B = la.solve_triangular(C, Yv.T, trans = 0, lower = True, check_finite = False)
        U, S, _ = sp.linalg.svd(B.T, full_matrices = False, check_finite = False)

        S = np.maximum(S**2 - v, 0.0)
        self.U = U
        self.S = S

    def _apply_precond(self,v):
         UTv = self.U.T@v
         w = UTv-(self.model.mu+self.S[self.s-1])*(UTv/(self.S+self.model.mu))
         return v-self.U@w
        