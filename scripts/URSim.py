import mujoco
import mujoco.viewer
import numpy as np
import time
from Integrators import DiscretizationMethods
from URModel import Model
from Force_Task_Space_PID_controller import PIDController
from TVLQR import TVLQRController
from TrajOpt import TrajOpt
from LQR import LQRController

class URSimTVLQR:
    def __init__(self):
        self.dt = 0.002
        self.model = Model(self.dt)
        self.integrator = DiscretizationMethods(self.model)

        self.Q = np.eye(2 * self.model.model.nv) * 5000
        self.R = np.eye(self.model.model.nu) * 0.001
        self.QN = np.eye(2 * self.model.model.nv) * 5000

        self.traj_opt = TrajOpt()
        self.tvlqr = TVLQRController(self.Q, self.R, self.QN)

    def initialize_sim(self, q_init, q_dot_init):
        self.model.data.qpos[:] = q_init
        self.model.data.qvel[:] = q_dot_init
        mujoco.mj_forward(self.model.model, self.model.data)
    
    def update_sim(self,q_curr,q_dot_curr):
        self.model.data.qpos[:] = q_curr
        self.model.data.qvel[:] = q_dot_curr
        mujoco.mj_forward(self.model.model, self.model.data)
    
    def run(self):
        
        q_init = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2])
        q_dot_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.initialize_sim(q_init, q_dot_init)
        
        q_des = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
        q_dot_des = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        start_time = 0.0
        end_time = 5.0
        total_steps = int((end_time - start_time) / self.dt) + 10

        print("Precomputing TVLQR Trajectory and Gains... (This might take a minute)")
        
        x_star_seq = []
        u_star_seq = []
        A_seq = []
        B_seq = []

        for k in range(total_steps):
            t = start_time + k * self.dt
            
            q_tar, q_dot_tar, q_ddot_tar = self.traj_opt.qubic_splines(
                t, q_init, q_dot_init, q_des, q_dot_des, start_time, end_time
            )
            x_tar = np.concatenate([q_tar, q_dot_tar])
            self.update_sim(x_tar[:6],x_tar[6:])
            u_tar = self.model.inv_dynamics(q_ddot_tar)
            A_c, B_c = self.model.get_linerazation_matrices(x_tar, u_tar)
            A_d, B_d = self.model.discritize_matrices(A_c, B_c)
            x_star_seq.append(x_tar)
            u_star_seq.append(u_tar)
            A_seq.append(A_d)
            B_seq.append(B_d)

        self.tvlqr.compute_gains(A_seq, B_seq)
        x_final = x_star_seq[-1]
        self.update_sim(x_final[:6],np.zeros(6))
        u_final = self.model.data.qfrc_bias.copy()
        A_final,B_final = self.model.get_linerazation_matrices(x_final, u_final)
        A_final,B_final = self.model.discritize_matrices(A_final,B_final)



        self.lqr = LQRController(
            Q= np.eye(2 * self.model.model.nv) * 10,
            R=np.eye(self.model.model.nu) * 0.1,
            QN=np.eye(2 * self.model.model.nv),
            x_bar=x_final,
            u_bar=u_final
        )
        self.lqr.compute_gains(A_final, B_final)
        
        self.initialize_sim(q_init, q_dot_init)
        step_index = 0
        t = 0
        with mujoco.viewer.launch_passive(self.model.model, self.model.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = np.array([1., 0., 1.])
            viewer.cam.distance = 2.
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -15

            while viewer.is_running():
                step_start = time.time()
                q_actual = np.copy(self.model.data.qpos)
                q_dot_actual = np.copy(self.model.data.qvel)
                x_state = np.concatenate([q_actual, q_dot_actual])

                if step_index < total_steps:
                    x_ref = x_star_seq[step_index]
                    u_bar = u_star_seq[step_index]
                    u_new = self.tvlqr.compute_control(step_index, x_state, x_ref, u_bar)

                else:
                    u_final = self.model.data.qfrc_bias.copy()
                    self.lqr.u_bar = u_final
                    # u_new = self.tvlqr.compute_control(step_index, x_state, x_final, u_final)
                    A_final,B_final = self.model.get_linerazation_matrices(x_final, u_final)
                    A_final,B_final = self.model.discritize_matrices(A_final,B_final)
                    self.lqr.compute_gains(A_final,B_final)
                    u_new = self.lqr.compute_control(x_state,x_final)

                    u_new = self.model.clip_control(u_new)
                    
                self.model.data.ctrl[:] = u_new
                mujoco.mj_step(self.model.model, self.model.data)

                step_index += 1
                t+=self.dt
                viewer.sync()
                
                time.sleep(max(0, self.dt - (time.time() - step_start)))


    def check_tvlqr_dynamics(self):
        print("--- Starting TVLQR Dynamics Check (Pure RK4) ---")
        
        q_init = np.array([0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, 0.0])
        q_des = np.array([-np.pi / 2, -np.pi / 2, 0.0, 0.0, np.pi / 2, 0.0])
        q_dot_init = np.zeros(6)
        q_dot_des = np.zeros(6)

        self.model.data.qpos[:] = q_init
        self.model.data.qvel[:] = q_dot_init
        self.model.data.ctrl[:] = self.model.data.qfrc_bias.copy()
        mujoco.mj_forward(self.model.model, self.model.data)

        start_time = 0.0
        end_time = 10.0
        sim_time = 20.0 
        
        traj_steps = int((end_time - start_time) / self.dt)
        sim_steps = int(sim_time / self.dt)

        # ==========================================
        # PHASE 1: PRECOMPUTATION
        # ==========================================
        print("Precomputing Trajectory and Gains...")
        x_star_seq = []
        u_star_seq = []
        A_seq = []
        B_seq = []

        for k in range(traj_steps):
            t = start_time + k * self.dt
            
            q_tar, q_dot_tar, q_ddot_tar = self.traj_opt.qubic_splines(
                t, q_init, q_dot_init, q_des, q_dot_des, start_time, end_time
            )
            x_tar = np.concatenate([q_tar, q_dot_tar])
            self.update_sim(x_tar[:6],x_tar[6:])
            u_tar = self.model.inv_dynamics(q_ddot_tar)
            # u_tar = self.model.clip_control(u_tar)
            
            A_c, B_c = self.model.get_linerazation_matrices(x_tar, u_tar)
            A_d, B_d = self.model.discritize_matrices(A_c, B_c)
            
            x_star_seq.append(x_tar)
            u_star_seq.append(u_tar)
            A_seq.append(A_d)
            B_seq.append(B_d)
            # Add this debug print in your precomputation loop
            # self.update_sim(x_tar[:6], x_tar[6:])
            print("gravity before linearization:", self.model.get_gravity_vector())
            A, B = self.model.get_linerazation_matrices(x_tar, u_tar)
            print("B matrix norm:", np.linalg.norm(B))  # will be identical every iteration

        self.tvlqr.compute_gains(A_seq, B_seq)
        x_final = x_star_seq[-1]
        self.update_sim(x_final[:6],np.zeros(6))

        u_final = self.model.data.qfrc_bias.copy()
        A_final,B_final = self.model.get_linerazation_matrices(x_final, u_final)



        self.lqr = LQRController(
            Q= np.eye(2 * self.model.model.nv) * 300,
            R=np.eye(self.model.model.nu) * 0.5,
            QN=self.tvlqr.P_seq[-1],
            x_bar=self.model.data.qpos.copy(),
            u_bar=u_final
        )
        self.lqr.compute_gains(A_final, B_final)
        print("Precomputation Complete.")
        self.initialize_sim(q_init,q_dot_init)

        # ==========================================
        # PHASE 2: PURE MATH SIMULATION (RK4)
        # ==========================================
        print("Simulating mathematically...")
        
        # Initialize our mathematical state vector
        x = np.concatenate([q_init, q_dot_init])
        
        x_hist = []
        u_hist = []

        for step_index in range(sim_steps):
            t = step_index * self.dt
            
            # Sync MuJoCo model to our RK4 state so gravity/inertia are correct for dynamics()
            

            # Get reference state and feedforward torque
            if step_index < traj_steps:
                x_ref = x_star_seq[step_index]
                u_bar = u_star_seq[step_index]
                u_new = self.tvlqr.compute_control(step_index, x, x_ref, u_bar)

          
            else:
                u_final = self.model.data.qfrc_bias.copy()
                u_new = self.tvlqr.compute_control(step_index, x, x_final, u_final)

            # TVLQR Control
            u_new = self.model.clip_control(u_new)


            # Record
            x_hist.append(x.copy())
            u_hist.append(u_new.copy())

            x = self.integrator.RungeKutta4thIntegration(x, u_new, self.dt)
            self.update_sim(x[:6],x[6:])

        x_hist = np.array(x_hist)
        u_hist = np.array(u_hist)

        # ==========================================
        # PHASE 3: VISUALIZE
        # ==========================================
        print("Simulation Complete. Generating plots...")
        self.model.visualize(x_hist, u_hist, self.dt, name='tvlqr_rk4_check')


def main():
    sim = URSimTVLQR()
    sim.run()

if __name__ == "__main__":
    main()