import numpy as np
import scipy as sp
import scipy.linalg as la

class RSN():
    """Implementation of RSN from "RSN: Randomized Subspace Newton" by Gower et al. 2019
    """
    def __init__(self,
                model,
                s):
        """Initialize RSN

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            s (int): Sketch size
        """
        self.model = model
        self.s = s
        self.is_sparse = sp.sparse.issparse(self.model.Atr)

    def step(self):
        """Perform a single step of RSN
        """
        g = self.model.get_grad(np.arange(self.model.ntr))        

        # Generate sketching matrix
        S = np.random.choice(self.model.p, self.s, replace = False)

        # The following steps work because of the structure of the sketching matrix and Hessian
        g_S = g[S]
        A_S = self.model.Atr[:, S]
        D = self.model.get_hessian_diag(np.arange(self.model.ntr))

        if self.is_sparse:
            D = sp.sparse.diags([D], [0])
            L = la.cholesky(A_S.T @ (D * A_S) + self.model.mu * sp.identity(self.s), lower = True)
        else:
            L = la.cholesky(A_S.T @ (A_S * D[:, np.newaxis]) + self.model.mu * sp.identity(self.s), lower = True)

        Linv_g_S = la.solve_triangular(L, g_S, trans = 0, lower = True, check_finite = False)
        LTinv_Linv_g_S = -la.solve_triangular(L, Linv_g_S, trans = 1, lower = True, check_finite = False)

        dir = np.zeros(self.model.p)
        dir[S] = LTinv_Linv_g_S
        t = self._line_search(dir, LTinv_Linv_g_S, S, g_S)
        self.model.w += t * dir

    def _line_search(self, dir, lambd, S, g_S):
        """Perform a line search to find the optimal step size.
        Algorithm 3 in supplementary material of Gower et al. 2019
        """
        a = 0.
        b = 1.
        la = np.inner(lambd, g_S)
        epsilon = np.abs(la) * 0.05
        lb = np.inner(lambd, self.model.get_grad(np.arange(self.model.ntr), v = self.model.w + b * dir)[S])

        while lb < -epsilon:
            a = b
            b *= 2 
            lb = np.inner(lambd, self.model.get_grad(np.arange(self.model.ntr), v = self.model.w + b * dir)[S])

        t = b
        lt = lb
        eps = np.finfo(b - a).eps

        while np.abs(lt) > epsilon and b - a > 10 * eps:
            if lt < 0:
                a = t
            else:
                b = t

            t = (a + b) / 2
            lt = np.inner(lambd, self.model.get_grad(np.arange(self.model.ntr), v = self.model.w + t * dir)[S])

        return t