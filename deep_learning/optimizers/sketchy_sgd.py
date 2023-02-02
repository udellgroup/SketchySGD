import torch
from torch.optim import Optimizer
from utils import normalization

class SketchySGD(Optimizer):
    """Implements SketchySGD. We assume that there is only one parameter group to optimize.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rank (int): sketch rank
        rho (float): regularization
        lr (float): learning rate
        weight_decay (float): weight decay parameter
        hes_update_freq (int): how frequently we should update the Hessian approximation
        proportional (bool): option to maintain lr to rho ratio, even when lr decays
        device (torch.device): device upon which we perform Hessian approximation updates
    """
    def __init__(self, params, rank = 100, rho = 0.1, lr = 0.01, weight_decay = 0.0, hes_update_freq = 1, proportional = False, device = "cpu"):
        # initialize the optimizer    
        defaults = dict(rank = rank, rho = rho, lr = lr, weight_decay = weight_decay, 
                        hes_update_freq = hes_update_freq, proportional = proportional, device = device)
        self.rank = rank
        self.hes_update_freq = hes_update_freq
        self.proportional = proportional
        self.ratio = rho / lr
        self.hes_iter = 0
        self.device = device
        self.U = None
        self.S = None
        super(SketchySGD, self).__init__(params, defaults)
         
    def step(self):
        # update Hessian approximation, if needed
        if self.hes_iter % self.hes_update_freq == 0:
            params = []
            grads = []

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        params.append(p)
                        grads.append(p.grad)

            # update Hessian and sketch
            self.update_hessian(params, grads)

        # one step update
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']

            # Adjust rho to be proportional to lr, if necessary
            if self.proportional:
                rho = lr * self.ratio
            else:
                rho = group['rho']

            # compute gradient as a long vector
            g = torch.cat([p.grad.view(-1) for p in group['params'] if p.grad is not None]) # only get gradients if they exist!
            # calculate the search direction by Nystrom sketch and solve
            UTg = torch.mv(self.U.t(), g) 
            g_new = torch.mv(self.U, (self.S + rho).reciprocal() * UTg) + g / rho - torch.mv(self.U, UTg) / rho
            
            ls = 0
            # update model parameters
            for p in group['params']:
                if p.grad is not None:
                    gp = g_new[ls:ls+torch.numel(p)].view(p.shape)
                    ls += torch.numel(p)
                    p.data.add_(-lr * (gp + weight_decay * p.data)) # use weight decay (not same as L2 reg.)
        
        self.hes_iter += 1

    def update_hessian(self, params, gradsH):
        # Check backward was called with create_graph set to True
        for i, grad in enumerate(gradsH):
            if grad.grad_fn is None:
                raise RuntimeError('Gradient tensor {:} does not have grad_fn. When calling\n'.format(i) +
                           '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                           '\t\t\t  set to True.')

        shift = 0.001
        # store random gaussian vector to a matrix
        test_matrix = []
        # Hessian vector product
        hv_matrix = []

        for i in range(self.rank):
            # generate gaussian random vector
            v = [torch.randn(p.size()).to(self.device) for p in params]
            # normalize
            v = normalization(v)
            # zero vector to store the shape
            hv_add = [torch.zeros(p.size()).to(self.device) for p in params]

            # calculate the Hessian vector product
            hv = torch.autograd.grad(gradsH, params, grad_outputs=v,only_inputs=True,retain_graph=True)
            # add initial shift
            for i in range(len(hv)):
                hv_add[i].data = hv[i].data.add_(hv_add[i].data)    
                hv_add[i].data = hv_add[i].data.add_(v[i].data * torch.tensor(shift)) 
            
            # reshape the Hessian vector product into a long vector
            hv_ex = torch.cat([gi.reshape(-1) for gi in hv_add])
            # reshape the random vector into a long vector
            test_ex = torch.cat([gi.reshape(-1) for gi in v])
            
            # append long vectors into a large matrix
            hv_matrix.append(hv_ex)
            test_matrix.append(test_ex)

        # assemble the large matrix
        hv_matrix_ex = torch.column_stack(hv_matrix)
        test_matrix_ex = torch.column_stack(test_matrix)
        # calculate Omega^T * A * Omega for Cholesky
        choleskytarget = torch.mm(test_matrix_ex.t(), hv_matrix_ex)
        # perform Cholesky, if fails, do eigendecomposition
        # the new shift is the abs of smallest eigenvalue (negative) plus the original shift
        try:
            C_ex = torch.linalg.cholesky(choleskytarget)
        except:
            # eigendecomposition, eigenvalues and eigenvector matrix
            eigs, eigvectors = torch.linalg.eigh(choleskytarget)
            shift = shift + torch.abs(torch.min(eigs))
            # add shift to eigenvalues
            eigs = eigs + shift
            # put back the matrix for Cholesky by eigenvector * eigenvalues after shift * eigenvector^T 
            C_ex = torch.linalg.cholesky(torch.mm(eigvectors, torch.mm(torch.diag(eigs), eigvectors.T)))
        
        # triangular solve
        B_ex = torch.linalg.solve_triangular(C_ex, hv_matrix_ex, upper = False, left = False)
        # SVD
        U, S, V = torch.linalg.svd(B_ex, full_matrices = False)
        self.U = U
        self.S = torch.max(torch.square(S) - torch.tensor(shift), torch.tensor(0.0))