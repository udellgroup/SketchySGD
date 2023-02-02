import numpy as np

class SGD():
    def __init__(self,
                model,
                eta):
        self.model = model
        self.eta = eta
        
    def step(self, indices):
        g = self.model.get_grad(indices)
        self.model.w -= self.eta * g