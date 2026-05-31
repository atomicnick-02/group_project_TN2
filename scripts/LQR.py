import numpy as np

class LQRController:
    def __init__(self, Q, R, QN=None, x_bar=None, u_bar=None, max_iter=2000, tol=1e-1):
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.QN = np.array(QN) if QN is not None else self.Q.copy()
        self.u_bar = u_bar
        self.x_bar = x_bar
        self.max_iter = max_iter
        self.tol = tol
        self.K = None
        self.P = None

    def compute_gains(self, A, B):
        P = self.QN.copy()
        error = np.inf
        iterations = 0
        while iterations < self.max_iter and error > self.tol:
            tmp1 = self.R + B.T @ P @ B
            tmp2 = B.T @ P @ A
            K = np.linalg.solve(tmp1, tmp2)
            tmp = A - B @ K
            P_new = self.Q + K.T @ self.R @ K + tmp.T @ P @ tmp
            error = np.linalg.norm(P - P_new)
            P = P_new
            iterations += 1
        self.K = K
        self.P = P
        return

    def compute_control(self, x_actual, x_target):
        noise_eps = 1e-2
        N = x_actual.shape[0]
        noise = np.random.uniform(-noise_eps, noise_eps,size=N)
        delta_x = x_actual - x_target + noise
        print(delta_x)
        delta_u = -self.K @ delta_x
        print(delta_u)

        u_new = self.u_bar + delta_u
        return u_new