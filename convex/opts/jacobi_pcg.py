import numpy as np
import scipy as sp
from .pcg import PCG

class JacobiPCG(PCG):
    def __init__(self, model):
        super().__init__(model, None)

    def _construct_precond(self):
        if sp.sparse.issparse(self.model.Atr):
            col_norms_sq = np.array(self.model.Atr.power(2).sum(axis=0))[0]
        else:
            col_norms_sq = np.sum(self.model.Atr**2, axis = 0)

        if self.model.fit_intercept:
            self.d = 1/self.model.ntr * col_norms_sq + self.model.mu
            self.d[0] = 1 # Correct for bias
        else:
            self.d = 1/self.model.ntr * col_norms_sq + self.model.mu

    def _apply_precond(self, v):
        return v/self.d