from URModel import Model

class DiscretizationMethods:
    def __init__(self, model: Model):
        self.model = model

    def RungeKutta4thIntegration(self, x, u, dt=0.05):
        f1 = self.model.dynamics(x, u)
        f2 = self.model.dynamics(x + (f1 * (dt / 2)), u)
        f3 = self.model.dynamics(x + (f2 * (dt / 2)), u)
        f4 = self.model.dynamics(x + f3 * dt, u)
        return x + (dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)

