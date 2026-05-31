import numpy as np

class TrajOpt:
    
    def __init__(self):
        pass

    def qubic_splines(self, t, q_start, q_dot_start, q_end, q_dot_end, t_start, t_end):
        total_time = t_end - t_start
        c0 = q_start
        c1 = q_dot_start
        c2 = (3 * q_end) / (np.square(total_time)) - (3 * q_start) / (np.square(total_time)) - (
                2 * q_dot_start) / total_time - q_dot_end / total_time
        c3 = - (2 * q_end) / (np.pow(total_time, 3)) + (2 * q_start) / (
            np.pow(total_time, 3)) + q_dot_start / np.square(total_time) + q_dot_end / np.square(total_time)

        q = c3 * np.pow(t, 3) + c2 * np.square(t) + c1 * t + c0
        q_dot = 3 * c3 * np.square(t) + 2 * c2 * t + c1
        q_ddot = 6 * c3 * t + 2 * c2

        if t > t_end:
            q = c3 * np.pow(t_end, 3) + c2 * np.square(t_end) + c1 * t_end + c0
            q_dot = 3 * c3 * np.square(t_end) + 2 * c2 * t_end + c1
            q_ddot = 6 * c3 * t_end + 2 * c2

        return q, q_dot, q_ddot