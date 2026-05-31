import mujoco
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


class Model:
    def __init__(self,dt):
        self.path = "C:/Users/stavr/OneDrive/Documents/PersonalProject/FirstSim/MujocoTest/mujoco_menagerie/universal_robots_ur10e/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.path)
        self.data = mujoco.MjData(self.model)
        self.ee_name = "attachment_site"
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)
        self.dt = dt
        # self.model.opt.timestep = self.dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        self.tau_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.tau_max = self.model.actuator_ctrlrange[:, 1].copy()
        N = 2 * self.model.nv
        M = self.model.nu
        dfx = jax.jit(jax.jacfwd(self.jax_dynamics,0))
        self.df_dx = lambda x,u : dfx(x,u).reshape(N,N)
        dfu = jax.jit(jax.jacfwd(self.jax_dynamics,1))
        self.df_du = lambda  x,u : dfu(x,u).reshape(N,M)
    
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

    def get_gravity_vector(self):
        return self.data.qfrc_bias
    
    def get_linerazation_matrices(self,x_bar,u_bar):
        q_curr = jnp.asarray(x_bar[:6])
        q_dot_curr = jnp.asarray(x_bar[6:])
        # x_bar = jnp.concatenate([q_curr,q_dot_curr])
        # u_bar = jnp.asarray(self.get_gravity_vector())
        A = np.asarray(self.df_dx(x_bar,u_bar))
        B = np.asarray(self.df_du(x_bar,u_bar))
        return A, B
    
    def discritize_matrices(self,A,B):
        N = A.shape[0] 
        M = B.shape[1] 
        # M = np.zeros((N + M, N + M))
        # M[:N, :N] = A
        # M[:N, N:] = B
        # M = M * self.dt
        # exp_M = scipy.linalg.expm(M)
        # A_d = exp_M[:N, :N]
        # B_d = exp_M[:N, N:]
        A_d = np.eye(N,N) + A * self.dt
        B_d = B * self.dt
        return A_d, B_d

    
    def dynamics(self, x, u):
        # Sync MuJoCo so gravity/inertia are correct for this substep
        self.data.qpos[:] = x[:6]
        self.data.qvel[:] = x[6:]
        mujoco.mj_forward(self.model, self.data)
        
        M = self.get_inertia_matrix()
        gravity = self.get_gravity_vector()
        q_ddot = np.linalg.inv(M) @ (u - gravity)
        return np.concatenate([x[6:], q_ddot])
    
    def inv_dynamics(self,q_ddot):
        M = self.get_inertia_matrix()
        gravity = self.get_gravity_vector()
        u = M @ q_ddot + gravity
        return u


    def jax_dynamics(self,x,u):
        q = jnp.asarray(jnp.copy(x[:6]))
        q_dot = jnp.asarray(jnp.copy(x[6:]))
        # self.data.qpos[:] = np.asarray(q.copy())
        # self.data.qvel[:] = np.asarray(q_dot.copy())
        # mujoco.mj_forward(self.model, self.data)
        M = jnp.asarray(self.get_inertia_matrix())
        gravity = jnp.asarray(self.get_gravity_vector())
        q_ddot = jnp.linalg.inv(M) @ (u - gravity)
        x_dot = jnp.concatenate([q_dot, q_ddot])
        return x_dot
    
    def clip_control(self,u):
        u_new = np.clip(u, self.tau_min, self.tau_max)
        return u_new
    
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

    