

import mujoco
import mujoco.viewer
import time
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


class UR10Sim:
    def __init__(self):
        self.path = "C:/Users/stavr/OneDrive/Documents/PersonalProject/FirstSim/MujocoTest/mujoco_menagerie/universal_robots_ur10e/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.path)
        self.data = mujoco.MjData(self.model)
        self.ee_name = "attachment_site"
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)
        self.dt = 0.002
        # mujoco.mj_forward(self.model, self.data)
        # self.model.opt.timestep = self.dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4

        self.Q = np.eye(2 * self.model.nv) * 10
        self.R = np.eye(self.model.nu) * .1
        # self.R[5, 5] = 1000.0
        self.QN = np.eye(2 * self.model.nv)
        self.tau_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.tau_max = self.model.actuator_ctrlrange[:, 1].copy()
        q_init = np.array([0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, 0.0])
        q_dot_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def LQR(self, A, B, Q, R):
        N = Q.shape[1]
        M = R.shape[1]
        It = 0
        K = np.zeros((M, N))
        P = self.QN
        error = np.inf
        while It < 2000 and error > 1e-1:
            tmp1 = (R + B.T @ P @ B)
            tmp2 = B.T @ P @ A
            K = np.linalg.solve(tmp1, tmp2)
            tmp = A - B @ K
            P_new = Q + K.T @ R @ K + tmp.T @ P @ tmp
            error = np.linalg.norm(P-P_new)
            It+=1
            P = P_new
        return K

    def get_linearization_fast(self):
        N = 2 * self.model.nv
        M = self.model.nu

        dfx = jax.jit(jax.jacfwd(self.jax_dynamics,0))
        df_dx = lambda x,u : dfx(x,u).reshape(N,N)
        dfu = jax.jit(jax.jacfwd(self.jax_dynamics,1))
        df_du = lambda  x,u : dfu(x,u).reshape(N,M)
        q_curr = np.asarray(self.data.qpos.copy())
        q_dot_curr = np.asarray(self.data.qvel.copy())
        x_bar = jnp.concatenate([q_curr,q_dot_curr])
        u_bar = jnp.asarray(self.data.qfrc_bias.copy())
        A= np.asarray(df_dx(x_bar,u_bar))
        B= np.asarray(df_du(x_bar,u_bar))
        return A, B

    def sin_move(self, t, amplitude=.5, omega=np.pi / 2):
        q = amplitude * np.sin(omega * t)
        q_dot = amplitude * omega * np.cos(omega * t)
        q_ddot = - amplitude * np.square(omega) * np.sin(omega * t)
        return q, q_dot, q_ddot

    def get_jacobian(self):
        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self.ee_id)
        J = np.vstack([Jp, Jr])
        return J

    def get_inertia_matrix(self):
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def get_corialis_matrix(self):
        C = np.zeros((self.model.nv, self.model.nv))
        return C

    def get_gravity_vector(self):
        G = np.zeros((self.model.nv, 1))

        return G

    def qubic_splines(self, t, q_start, q_dot_start, q_end, q_dot_end, t_start, t_end, ):
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

    def dynamics(self, x, u):
        q = x[:6]
        q_dot = x[6:]
        self.data.qpos[:] = q
        self.data.qvel[:] = q_dot
        mujoco.mj_forward(self.model, self.data)
        M = self.get_inertia_matrix()
        q_ddot = np.linalg.inv(M) @ (u - self.data.qfrc_bias)
        x_dot = np.concatenate([q_dot, q_ddot])
        return x_dot


    def jax_dynamics(self,x,u):
        q = x[:6]
        q_dot = jnp.asarray(x[6:])
        M = jnp.asarray(self.get_inertia_matrix())
        q_ddot = jnp.linalg.inv(M) @ (u - jnp.asarray(self.data.qfrc_bias))
        x_dot = jnp.concatenate([q_dot, q_ddot])
        return x_dot

    def RK4(self, x, u, dt=0.05):
        f1 = self.dynamics(x, u)
        f2 = self.dynamics(x + (f1 * (dt / 2)), u)
        f3 = self.dynamics(x + (f2 * (dt / 2)), u)
        f4 = self.dynamics(x + f3 * dt, u)
        return x + (dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)



    def run(self):
        model = self.model
        data = self.data
        dt = model.opt.timestep
        t = 0
        # mujoco.mj_forward(self.model, self.data)
        q_init = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2])
        q_dot_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


        # mujoco.mj_forward(model, data)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = np.array([1., 0., 1.])
            viewer.cam.distance = 2.
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -15

            qmin = model.jnt_range[:, 0].copy()
            qmax = model.jnt_range[:, 1].copy()
            q_ref = np.zeros((qmin.shape[0],))
            q_ref[:] = (qmin[:] + qmax[:]) / 2

            Kp = np.diag([500.0, 500.0, 500.0, 500.0, 500.0, 500.0])
            Ki = np.diag([80.0, 80.0, 80.0, 80.0, 80.0, 80.0])
            Kd = np.diag([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

            integral_error = 0.0
            # mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = q_init
            self.data.qvel[:] = q_dot_init

            mujoco.mj_forward(self.model, self.data)

            A, B = self.get_linearization_fast()
            A = np.eye(A.shape[0],A.shape[1]) + A*dt
            B = B*dt
            K = self.LQR(A, B, self.Q, self.R)

            while viewer.is_running():

                step_start = time.time()
                qs = self.data.qpos
                qs_dot = q_dot_init.copy()
                qg = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
                qg_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                start_time = 0.0
                end_time = 2.0
                q_target, q_dot_target, q_ddot_target = self.qubic_splines(t, qs, qs_dot, qg, qg_dot, start_time,
                                                                           end_time)

                # Read actual states
                q_actual = np.copy(data.qpos)
                q_dot_actual = np.copy(data.qvel)
                x_state = np.concatenate([q_actual, q_dot_actual])
                x_ref = np.concatenate([q_target, q_dot_target])

                # Calculate errors
                # error = q_target - q_actual
                # error_dot = q_dot_target - q_dot_actual
                #
                # integral_error += error * dt
                # M = self.get_inertia_matrix()
                # desired_accel = q_ddot_target + (Kp @ error) + (Ki @ integral_error) + (Kd @ error_dot)

                u_bar =  data.qfrc_bias
                x = x_state.copy()
                noise_eps = 1e-1
                delta_x = x - x_ref + np.random.uniform(-noise_eps, noise_eps,size=(2*self.model.nv,))
                delta_u = -K @ delta_x
                u_new = u_bar + delta_u
                u_new = np.clip(u_new, self.tau_min, self.tau_max)
                self.data.ctrl[:] = u_new


                mujoco.mj_step(self.model, self.data)

                t += dt
                viewer.sync()

                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    def check_dynamics(self):
        q_init = np.array([0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, 0.0])
        q_target = np.array([-np.pi/2, -np.pi / 2, 0.0, 0.0, np.pi / 2, 0.0])
        q_dot_init = np.zeros(6)
        tau_min = self.model.actuator_ctrlrange[:, 0].copy()
        tau_max = self.model.actuator_ctrlrange[:, 1].copy()
        print("Control limits:", self.model.actuator_ctrlrange)
        self.data.qpos[:] = q_init
        self.data.qvel[:] = q_dot_init
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = self.data.qfrc_bias.copy()  # ← set ctrl before linearizing
        mujoco.mj_forward(self.model, self.data)
        u_test = self.data.qfrc_bias.copy()

        print("Equilibrium torque:", u_test)

        x_test = np.concatenate([q_init, q_dot_init])
        x_target = np.concatenate([q_target,q_dot_init])

        x = x_test.copy()
        u = u_test.copy()

        time = 10.0
        steps = int(time / self.dt)
        x_hist = []
        u_hist = []

        A,B = self.get_linearization_fast()
        A = np.eye(A.shape[0], A.shape[1]) + A * self.dt
        B = B * self.dt
        K = self.LQR(A,B,self.Q,self.R)
        for i in range(steps):
            self.data.qpos[:] = x[:6]
            self.data.qvel[:] = x[6:]
            mujoco.mj_forward(self.model, self.data)
            u_grav = self.data.qfrc_bias.copy()

            delta_x = x - x_target
            u_new = u_grav + (-K @ delta_x)
            u_new = np.clip(u_new, tau_min, tau_max)

            x = self.RK4(x, u_new, self.dt)
            x_hist.append(x.copy())
            u_hist.append(u_new.copy())




        x_hist = np.array(x_hist)
        u_hist = np.array(u_hist)

        self.visualize(x_hist, u_hist, self.dt)

    def visualize(self, x, u, dt, name='sim_results'):
        fig = plt.figure(figsize=(12, 8))
        ax = [None] * 2
        ax_labels_x = [
            "q1", "q2", "q3", "q4", "q5", "q6",
            "q1_dot", "q2_dot", "q3_dot", "q4_dot", "q5_dot", "q6_dot"
        ]
        ax_labels_u = ["tau1", "tau2", "tau3", "tau4", "tau5", "tau6"]

        ax[0] = fig.add_subplot(211)
        ax[1] = fig.add_subplot(212)

        # Plot state trajectories
        for i in range(6):
            ax[0].plot(
                [k * dt for k in range(x.shape[0])],
                x[:, i],
                label=ax_labels_x[i],
                linewidth=2,
            )
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("State")
        ax[0].set_title("State Trajectories")
        ax[0].legend()
        ax[0].grid(True)

        # Plot control trajectories
        for i in range(6):
            ax[1].plot(
                [k * dt for k in range(u.shape[0])],
                u[:, i],
                label=ax_labels_u[i],
                linewidth=2,
            )
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Control Input")
        ax[1].set_title("Control Inputs")
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig(str(name), dpi=300, bbox_inches='tight')
        plt.show()






def main():
    sim = UR10Sim()
    sim.run()


if __name__ == "__main__":
    main()



