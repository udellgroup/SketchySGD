import numpy as np
import scipy as sp
import scipy.linalg as la

class Logistic(): # Defines a logistic regression problem (with ridge parameter + intercept)
    def __init__(self,
                Atr,
                btr,
                Atst,
                btst,
                mu,
                fit_intercept = False):

        self.Atr = Atr
        self.btr = btr
        self.Atst = Atst
        self.btst = btst
        self.ntr = Atr.shape[0]
        self.ntst = Atst.shape[0]
        self.mu = mu # Ridge parameter
        self.fit_intercept = fit_intercept
        self.p = Atr.shape[1]
        if self.fit_intercept:
            self.w = np.zeros(self.p + 1)
            self.p += 1
        else:
            self.w = np.zeros(self.p)

    def get_losses(self):
        if self.fit_intercept: # Don't incorporate bias into regularization
            train_loss = 1/self.ntr * sum(np.log(1 + np.exp(-np.multiply(self.btr, self.Atr @ self.w[1:] + self.w[0])))) + self.mu/2 * np.linalg.norm(self.w[1:])**2
            test_loss = 1/self.ntst * sum(np.log(1 + np.exp(-np.multiply(self.btst, self.Atst @ self.w[1:] + self.w[0]))))
        else:
            train_loss = 1/self.ntr * sum(np.log(1 + np.exp(-np.multiply(self.btr, self.Atr @ self.w)))) + self.mu/2 * np.linalg.norm(self.w)**2 
            test_loss = 1/self.ntst * sum(np.log(1 + np.exp(-np.multiply(self.btst, self.Atst @ self.w))))

        return {'train_loss': train_loss, 'test_loss': test_loss}

    def get_acc(self):
        # Train accuracy
        y_hat = np.zeros(self.ntr)
        if self.fit_intercept:
            prob = 1/(1 + np.exp(-(self.Atr @ self.w[1:] + self.w[0])))
        else:
            prob = 1/(1 + np.exp(-self.Atr @ self.w))
        Jplus = np.argwhere(prob >= 0.5)
        Jminus = np.argwhere(prob < 0.5)
        y_hat[Jplus] = 1
        y_hat[Jminus] = -1
        class_err = 100 * np.count_nonzero(y_hat - self.btr) / self.ntr
        train_acc = 100 - class_err

        # Test accuracy
        y_hat = np.zeros(self.ntst)
        if self.fit_intercept:
            prob = 1/(1 + np.exp(-(self.Atst @ self.w[1:] + self.w[0])))
        else:
            prob = 1/(1 + np.exp(-self.Atst @ self.w))
        Jplus = np.argwhere(prob >= 0.5)
        Jminus = np.argwhere(prob < 0.5)
        y_hat[Jplus] = 1
        y_hat[Jminus] = -1
        class_err = 100 * np.count_nonzero(y_hat - self.btst) / self.ntst
        test_acc = 100 - class_err

        return {'train_acc': train_acc, 'test_acc': test_acc}

    # v is needed for algorithms such as SVRG that need gradients at locations other than the current iterate
    def get_grad(self, indices, v = None):
        n = indices.shape[0]
        X = self.Atr[indices,:]
        y = self.btr[indices]

        # If no input provided, just use current iterate for computing the gradient
        if v is None:
            v = self.w

        if self.fit_intercept:
            g_intermediate = np.divide(-y, 1 + np.exp(np.multiply(y, X @ v[1:] + v[0])))
            g = 1/n * np.concatenate((np.array([g_intermediate.sum()]), X.T @ g_intermediate)) + self.mu * np.concatenate((np.array([0]), v[1:]))
        else:
            g = 1/n * (X.T @ (np.divide(-y, 1 + np.exp(np.multiply(y, X @ v))))) + self.mu * v
        return g

    def get_hessian_diag(self, indices):
        n = indices.shape[0]
        X = self.Atr[indices,:]
        
        if self.fit_intercept:
            probs = 1/(1 + np.exp(-(X @ self.w[1:] + self.w[0])))
        else:
            probs = 1/(1 + np.exp(-X @ self.w))
        
        D2 = probs * (1 - probs)/n
        D2 = np.array(D2)
        return D2

    def get_rand_nys_appx(self, indices, rank):
        n = indices.shape[0]
        D2 = self.get_hessian_diag(indices)
        Omega = np.random.randn(self.p, rank)
        Omega = la.qr(Omega, mode='economic')[0]

        if self.fit_intercept:
            if sp.sparse.issparse(self.Atr):
                d = D2 ** (1/2)
                d_mat = sp.sparse.diags([d], [0])
                X = sp.sparse.hstack((sp.sparse.csr_matrix(d).T, d_mat * self.Atr[indices, :]))
            else:
                d = np.power(D2,1/2)
                X = np.column_stack((d, np.einsum('i,ij->ij', d, self.Atr[indices, :])))
        else:
            if sp.sparse.issparse(self.Atr):
                d = D2 ** (1/2)
                d = sp.sparse.diags([d], [0])
                X = d * self.Atr[indices, :]
            else:
                X = np.einsum('i,ij->ij', np.power(D2,1/2), self.Atr[indices, :])
        Y = X.T @ (X @ Omega)

        v = np.sqrt(rank)*np.spacing(np.linalg.norm(Y,2))
        Yv = Y+v*Omega
        Core = Omega.T@Yv
        try:
             C = np.linalg.cholesky(Core)
        except:
            eig_vals = la.eigh(Core,eigvals_only=True)
            v = v+np.abs(np.min(eig_vals))
            Core = Core+v*np.eye(rank)
            C = np.linalg.cholesky(Core)

        B = la.solve_triangular(C, Yv.T, trans = 0, lower = True, check_finite = False)
        U, S, _ = sp.linalg.svd(B.T, full_matrices = False, check_finite = False)
        S = np.maximum(S**2 - v, 0.0)

        return U, S