import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator

class AdaSGD():
    """Implementation of SGD with adaptive stepsize based on minibatch Hessian
    """    
    def __init__(self,
                model,
                update_freq,
                bh):
        """Initialize SGD

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            update_freq (dict): Update frequency for learning rate.
                Must contain (key, value) pair ('learning_rate', (int, 'minibatches'))
            bh: hessian batch size for computing learning rate
        """                
        self.model = model
        self.update_freq = update_freq
        self.bh = bh
        self.n_iters = 0
        
    def step(self, indices):
        """Perform a single step of SGD

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient
        """ 
        # Update the learning rate at appropriate frequency 
        if self.n_iters % self.update_freq['learning_rate'] == 0:
            h_indices = np.random.choice(self.model.ntr, self.bh, replace = False)
            self.eta = 1 / self._compute_eig(h_indices) * 1/2
        
        g = self.model.get_grad(indices)
        self.model.w -= self.eta * g
        self.n_iters += 1
    
    def _compute_eig(self, h_indices):
        """Estimate the largest eigenvalue of the preconditioned Hessian via subsampling

        Args:
            h_indices (ndarray): 1d array of row indices for subsampling training data

        Returns:
            float: Estimated largest eigenvalue
        """      
        lin_op = self._get_lin_op(self.model.get_hessian_diag(h_indices), 
                                self.model.Atr[h_indices, :])
        eig_val = sp.sparse.linalg.eigs(lin_op, k = 1, which = 'LM', return_eigenvectors = False)
        return np.real(eig_val[0])

    def _get_lin_op(self, D, A):
        def mv(x):
            return A.T @ (D * (A @ x)) + self.model.mu * x

        return LinearOperator((A.shape[1], A.shape[1]), 
                                matvec = lambda x: mv(x.reshape(-1)), 
                                rmatvec = lambda x: mv(x.reshape(-1)))