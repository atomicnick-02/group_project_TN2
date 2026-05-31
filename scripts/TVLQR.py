# import numpy as np

# class TVLQRController:
    
#     def __init__(self, Q, R, QN=None):
#         self.Q = np.array(Q)
#         self.R = np.array(R)
#         self.QN = np.array(QN) if QN is not None else self.Q.copy()
        
#         self.K_seq = []
#         self.P_seq = []

#     def compute_gains(self, Ak, Bk):
#         gains = []
#         P = self.QN
#         for k in range(self.K_steps - 2, -1, -1):
#             x_k = self.x_nom[k]
#             u_k = self.u_nom[k]
#             mat_inv = np.linalg.inv(self.R + Bk.T @ P @ Bk)
#             K_k = mat_inv @ Bk.T @ P @ Ak
#             P = self.Q + Ak.T @ P @ Ak - Ak.T @ P @ Bk @ K_k
#             gains.insert(0, K_k)
#             self.K_seq = np.copy(np.array([gains]))
#         return gains

#     def compute_control(self, step_index, x_actual, x_target, u_bar):
#         if step_index >= len(self.K_seq):
#             K = self.K_seq[-1]
#         else:
#             K = self.K_seq[step_index]
#         delta_x = x_actual - x_target
#         delta_u = -K @ delta_x
#         u_new = u_bar + delta_u
#         return u_new

import numpy as np
import jax.numpy as jnp

class TVLQRController:
    
    def __init__(self, Q, R, QN=None):
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.QN = np.array(QN) if QN is not None else self.Q.copy()
        self.K_seq = []
        self.P_seq = []

    def compute_gains(self, A_seq, B_seq):
        num_steps = len(A_seq)
        self.K_seq = [None] * num_steps
        self.P_seq = [None] * (num_steps + 1)
        self.P_seq[-1] = self.QN.copy()
        for k in range(num_steps - 1, -1, -1):
            A_k = A_seq[k]
            B_k = B_seq[k]
            P_next = self.P_seq[k + 1]
            tmp1 = jnp.asarray(self.R + B_k.T @ P_next @ B_k)
            tmp2 = jnp.asarray(B_k.T @ P_next @ A_k)
            K_k = jnp.linalg.solve(tmp1, tmp2)
            self.K_seq[k] = np.asarray(K_k)
            tmp = A_k - B_k @ K_k
            P_k = self.Q + (K_k.T @ self.R @ K_k) + (tmp.T @ P_next @ tmp)
            self.P_seq[k] = P_k
        return

    def compute_control(self, step_index, x_actual, x_target, u_bar):
        noise_eps = 1e-2
        N = x_actual.shape[0]
        noise = np.random.uniform(-noise_eps, noise_eps,size=N)
        if step_index >= len(self.K_seq):
            K = self.K_seq[-1]
        else:
            K = self.K_seq[step_index]
        delta_x = x_actual - x_target + noise
        delta_u = -K @ delta_x
        u_new = u_bar + delta_u 
        return u_new