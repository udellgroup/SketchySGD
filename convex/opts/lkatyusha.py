import numpy as np

class LKatyusha():
    def __init__(self,
                model,
                mu,
                L,
                bg,
                theta1 = None,
                theta2 = None,
                p = None):
        self.model = model
        self.theta1 = theta1
        self.theta2 = theta2
        self.mu = mu
        self.L = L
        self.sigma = self.mu / self.L

        if p is None:
            self.p = bg / self.model.ntr
        else:
            self.p = p

        if theta1 is None:
            self.theta1 = np.minimum(np.sqrt(2 * self.model.ntr * self.sigma/3), 0.5)
        else:
            self.theta1 = theta1
        if theta2 is None:
            self.theta2 = 0.5
        else:
            self.theta2 = theta2

        # Complete initialization
        self.eta = self.theta2 / ((1 + self.theta2) * self.theta1)
        self.y = self.model.w.copy()
        self.z = self.model.w.copy()
        self.x = None
        self.g_bar = self.model.get_grad(np.arange(self.model.ntr), self.y)

    def step(self, indices):
        self.x = self.theta1 * self.z + self.theta2 * self.y + (1 - self.theta1 - self.theta2) * self.model.w
        g = self.model.get_grad(indices, self.x) - self.model.get_grad(indices, self.y) + self.g_bar
        z_next = 1/(1 + self.eta * self.sigma) * (self.eta * self.sigma * self.x + self.z - self.eta/self.L * g)
        w_next = self.x + self.theta1 * (z_next - self.z)

        if np.random.rand() < self.p:
            self.y = self.model.w.copy()
            self.g_bar = self.model.get_grad(np.arange(self.model.ntr), self.y)

        self.z = z_next
        self.model.w = w_next