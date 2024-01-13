import numpy as np
from abc import ABC, abstractmethod

class PCG(ABC):
    def __init__(self, model, s):
        self.model = model
        self.s = s
        self.initialized = False

    @abstractmethod
    def _construct_precond(self):
        pass

    @abstractmethod
    def _apply_precond(self):
        pass

    def _init_pcg(self):
        self.r0 = 1/self.model.ntr*self.model.Atr.T@(self.model.btr-self.model.Atr@self.model.w)-self.model.mu*self.model.w
        self.z0 = self._apply_precond(self.r0)
        self.p0 = np.copy(self.z0)

    def step(self):
        if not self.initialized:
            self._construct_precond()
            self._init_pcg()
            self.initialized = True

        v = 1/self.model.ntr*self.model.Atr.T@(self.model.Atr@self.p0)+self.model.mu*self.p0
        alpha = (self.r0.T@self.z0)/(v.T@self.p0)
        self.model.w += alpha*self.p0
        r = self.r0-alpha*v
        z = self._apply_precond(r)
        beta = (z.T@r)/(self.z0.T@self.r0)
        self.p0 = z+beta*self.p0
        self.r0 = r 
        self.z0 = z 