import numpy as np

class PIDController:

    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt
        self.integral_error = np.zeros(self.Kp.shape[0])

    def compute_acceleration(self, q_actual, q_target, q_dot_actual, q_dot_target, q_ddot_target=None):
        error = q_target - q_actual
        error_dot = q_dot_target - q_dot_actual
        self.integral_error += error * self.dt
        desired_accel = q_ddot_target + (self.Kp @ error) + (self.Ki @ self.integral_error) + (self.Kd @ error_dot)
        return desired_accel

    def compute_torque(self, q_actual, q_target, q_dot_actual, q_dot_target, M_matrix, qfrc_bias, q_ddot_target=None):
        desired_accel = self.compute_acceleration(q_actual, q_target, q_dot_actual, q_dot_target, q_ddot_target)
        tau = (M_matrix @ desired_accel) + qfrc_bias
        return tau