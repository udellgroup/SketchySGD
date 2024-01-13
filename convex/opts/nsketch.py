import numpy as np
import scipy as sp
import scipy.linalg as la
from scipy.sparse import csr_matrix

class NSketch():
    """Implementation of Newton Sketch from "Newton Sketch: A Near Linear-time Optimization Algorithm
    with Linear-quadratic Convergence" by Pilanci and Wainwright 2017
    """
    def __init__(self,
                 model,
                 s):
        """Initialize NSketch

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            s (int): Sketch size
        """
        self.model = model
        self.s = s
        self.is_sparse = sp.sparse.issparse(self.model.Atr)

        # Line search parameters
        self.a = 0.1
        self.b = 0.5

    def _generate_embedding(self):
        n = self.model.ntr
        zeta = min(self.s, 8)
        # rows = np.random.choice(range(r), zeta * s) # Faster choice that allows for repeated indices
        rows = np.random.rand(n,self.s).argsort(axis = -1)
        rows = rows[:, :zeta].reshape(-1)
        cols = np.kron(range(n), np.ones(zeta))
        signs = np.sign(np.random.uniform(0, 1.0, len(cols)) - 0.5)
        Omega = csr_matrix((signs/np.sqrt(zeta), (rows, cols)), shape = (self.s, n))
        return Omega

    def step(self):
        """Perform a single step of Newton Sketch
        """
        g = self.model.get_grad(np.arange(self.model.ntr))

        # Generate sketching matrix
        # S = 1/np.sqrt(self.s) * np.random.randn(self.s, self.model.ntr)
        S = self._generate_embedding()

        # Get Hessian square root (without regularization)
        D = self.model.get_hessian_diag(np.arange(self.model.ntr)) ** (1/2)
        if self.is_sparse:
            D = sp.sparse.diags([D], [0])
            H_sqrt = D * self.model.Atr
        else:
            H_sqrt = self.model.Atr * D[:, np.newaxis]

        # Get sketch of Hessian square root
        Y = S @ H_sqrt

        # Compute descent direction via Woodbury formula
        # L = la.cholesky(Y @ Y.T + self.model.mu * sp.identity(self.s),
        #                  lower = True) # Should deliver speedups when self.s << self.model.p

        # Yg = Y @ g
        # Linv_Yg = la.solve_triangular(L, Yg, trans = 0, lower = True, check_finite = False)
        # LTinv_Linv_Yg = la.solve_triangular(L, Linv_Yg, trans = 1, lower = True, check_finite = False)
        # YT_LTinv_Linv_Yg = Y.T @ LTinv_Linv_Yg
        # dir = (-g + YT_LTinv_Linv_Yg) / self.model.mu

        Yg = Y @ g
        dir = (-g + Y.T @ la.solve(Y @ Y.T + self.model.mu * sp.identity(self.s),
                                    Yg, assume_a = 'pos', check_finite = False)) / self.model.mu
        
        t = self._line_search(dir, g)
        self.model.w += t * dir

    def _line_search(self, dir, g):
        """Perform Armijo line search to find optimal step size
        """
        f = self.model.get_losses(both = False)['train_loss']
        lambd = np.inner(g, dir)

        t = 1.
        while self.model.get_losses(v = self.model.w + t * dir, both = False)['train_loss'] \
                        > f + self.a * t * lambd:
            t *= self.b
        return t