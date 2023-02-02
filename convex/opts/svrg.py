import numpy as np

class SVRG():
    def __init__(self,
                model,
                eta,
                update_freq):
        self.model = model
        self.eta = eta
        self.update_freq = update_freq
        self.w_tilde = None
        self.g_bar = None
        self.n_iters = 0

    def step(self, indices):
        # Update snapshot if needed
        if self.n_iters % self.update_freq['snapshot'] == 0:
            self.w_tilde = self.model.w.copy()
            self.g_bar = self.model.get_grad(np.arange(self.model.ntr))
        
        g = self.model.get_grad(indices)
        g_tilde = self.model.get_grad(indices, self.w_tilde)
        self.model.w -= self.eta * (g - g_tilde + self.g_bar) # SVRG update
        self.n_iters += 1