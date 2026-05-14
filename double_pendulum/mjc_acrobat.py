import numpy as np
import os
import jax
import jax.numpy as jnp
from dm_control import suite
import types


# Fixed Variables
GRAVITY = 9.81  # gravity in m/s^2
mass = [1, 1]
length = [0.05, 0.049]
damping = [0.0, 0.0]
coulomb_fric = [0.0, 0.0]

l1, l2 = length
m1, m2 = mass
b1, b2 = damping
mu1, mu2 = coulomb_fric


Q = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))
Qfin = Q
R = jnp.diag(jnp.array([1, 1])) * 0.01
max_torque = 1  # Physical Hard Stop

# Read the optimal trajectory from file
results_dir = os.path.join(os.path.dirname(__file__), "results")
x_ref = np.loadtxt(os.path.join(results_dir, "trajectory.csv"), delimiter=",", skiprows=1).T
u_ref = np.loadtxt(os.path.join(results_dir, "inputs.csv"), delimiter=",", skiprows=1).T
K_matrix = np.load(os.path.join(results_dir, "K_matrix.npy"))

# ─────────────────────────────────────────────
# Load environment
# ─────────────────────────────────────────────
env = suite.load(
    domain_name="acrobot",
    task_name="swingup",
    task_kwargs={"time_limit": 20.0},
    visualize_reward=True,
)

action_spec = env.action_spec()
obs_spec    = env.observation_spec()

print("Action spec:", action_spec)          # shape=(1,) continuous torque
print("Obs spec:   ", obs_spec)             # orientations + velocities

# ─────────────────────────────────────────────
# Force deterministic reset to bottom position
# ─────────────────────────────────────────────
original_init = env.task.initialize_episode

def exact_bottom_init(self, physics):
    # Call the original initialization first to handle any internal setup
    original_init(physics)
    
    # Overwrite the joint positions and velocities to be exactly zeroed/down
    # shoulder = np.pi (straight down), elbow = 0.0 (straight)
    physics.named.data.qpos['shoulder'] = np.pi
    physics.named.data.qpos['elbow'] = 0.0
    
    # Kill any initial velocities
    physics.named.data.qvel['shoulder'] = 0.0
    physics.named.data.qvel['elbow'] = 0.0

# Bind this custom method to the existing task instance
env.task.initialize_episode = types.MethodType(exact_bottom_init, env.task)

def flatten_obs(timestep):
    """
    dm_control obs is a dict. Flatten to a single vector.
    For acrobot/swingup:
      - orientations: [cos(θ1), sin(θ1), cos(θ2), sin(θ2)]  shape (4,)
      - velocity:     [θ̇1, θ̇2]                               shape (2,)
    Total: 6-dim state vector.
    """
    return np.concatenate([v.flatten() for v in timestep.observation.values()])


class OptimalController:
    """
    Time-Varying LQR controller for the acrobot swingup task.
    
    Uses pre-computed reference trajectory (x_ref, u_ref) and gain matrices (K)
    to track the trajectory with TRACKING and RECOVERY modes.
    """
    
    def __init__(self, x_ref, u_ref, K_matrix, max_torque=0.07, dt=0.05):
        """
        Initialize the optimal controller.
        
        Args:
            x_ref: Reference state trajectory, shape (nx, N)
            u_ref: Reference input trajectory, shape (nu, N)
            K_matrix: Pre-computed gain matrices, shape (N-1, nu, nx)
            max_torque: Maximum torque limit for clipping
            dt: Control timestep
        """
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.K_matrix = K_matrix
        self.max_torque = max_torque
        self.dt = dt
        
        self.nx = x_ref.shape[0]
        self.nu = u_ref.shape[0]
        self.max_idx = x_ref.shape[1] - 1
        
        # Tracking state
        self.current_idx = 0
        self.mode = "TRACKING"
        self.deviation_threshold = 10.0
        
    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _to_feature_space(self, x):
        """
        Convert state [q1, q2, v1, v2] to feature space for distance calculation.
        Feature space: [cos(q1), sin(q1), cos(q2), sin(q2), 0.1*v1, 0.1*v2]
        """
        v_scale = 0.1
        if x.ndim == 1:
            p0, p1 = x[0], x[1]
            v = x[2:] * v_scale
            return np.concatenate(([np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1)], v))
        else:
            p0, p1 = x[0, :], x[1, :]
            v = x[2:, :] * v_scale
            return np.vstack((np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1), v))
    
    def _get_closest_indices(self, x, k=1):
        """
        Get k closest trajectory indices to state x in feature space.
        
        Returns:
            indices: Indices of k closest points
            distances: Euclidean distances in feature space
        """
        feat_x = self._to_feature_space(x)
        feat_ref = self._to_feature_space(self.x_ref)
        
        diff = feat_ref - feat_x.reshape(-1, 1)
        dists = np.linalg.norm(diff, axis=0)
        
        sorted_indices = np.argsort(dists)
        return sorted_indices[:k], dists[sorted_indices[:k]]
    
    def _get_nearest_idx(self, x):
        """Get the nearest trajectory index to state x."""
        idx, _ = self._get_closest_indices(x, k=1)
        return idx[0]
    
    def _get_interpolated_gain(self, x, k=5):
        """
        Get interpolated gain matrix using weighted averaging of k nearest points.
        
        Weights are inversely proportional to distance in feature space.
        """
        indices, dists = self._get_closest_indices(x, k=k)
        eps = 1e-6
        w = 1.0 / (dists + eps)
        weights = w / np.sum(w)
        
        # Clip indices to valid range for K matrix
        indices_clipped = np.clip(indices, 0, len(self.K_matrix) - 1)
        K_selected = self.K_matrix[indices_clipped]
        
        K_weighted = np.sum(weights[:, None, None] * K_selected, axis=0)
        return K_weighted
    
    def _check_tracking_status(self, x_current):
        """
        Check if system is tracking well or needs recovery.
        
        Returns:
            distance to target: Euclidean distance in feature space
            holding: Boolean indicating if at end of trajectory
        """
        target_state = self.x_ref[:, self.current_idx].reshape(-1, 1)
        _, dists = self._get_closest_indices(x_current, k=1)
        dist_to_target = dists[0]
        
        holding = (self.current_idx >= self.max_idx)
        
        return dist_to_target, holding
    
    def get_control(self, x_current):
        """
        Compute control input using TVLQR.
        
        Modes:
            - TRACKING: Follow reference trajectory, advance along trajectory
            - RECOVERY: Weighted interpolation of K gains, find nearest trajectory point
        
        Args:
            x_current: Current state [q1, q2, q1_dot, q2_dot], shape (4,)
        
        Returns:
            u_total: Control input [u1, u2], clipped to hardware limits
        """
        # Check tracking status
        dist_to_target, holding = self._check_tracking_status(x_current)
        
        if dist_to_target > self.deviation_threshold or holding:
            self.mode = "RECOVERY"
            # Find nearest trajectory point
            self.current_idx = self._get_nearest_idx(x_current)
        else:
            self.mode = "TRACKING"
        
        # Get reference state and input
        idx = self.current_idx
        x_des = self.x_ref[:, idx]
        u_des = self.u_ref[:, idx]
        
        # Compute gain
        if self.mode == "TRACKING":
            K_gain = self.K_matrix[idx] if idx < len(self.K_matrix) else self.K_matrix[-1]
            # Advance tracking index
            self.current_idx = min(idx + 1, self.max_idx)
        else:  # RECOVERY
            K_gain = self._get_interpolated_gain(x_current, k=5)
        
        # Compute error
        error = np.array(x_current - x_des)
        # Wrap angles
        error[0] = self._wrap_angle(error[0])
        error[1] = self._wrap_angle(error[1])
        
        # Feedback control
        u_feedback = -K_gain @ error
        u_total = u_des + u_feedback.flatten()
        
        # Clip to hardware limits
        u_total = np.clip(u_total, -self.max_torque, self.max_torque)
        
        return u_total
    
    def reset(self):
        """Reset the controller state."""
        self.current_idx = 0
        self.mode = "TRACKING"

def random_policy(timestep):
    """A random policy for testing.
    0 - shoulder
    1 - elbow
    """
    
    # zero the shoulder (first joint) and randomize
    random_action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
    # random_action[1] = max_torque 
    # random_action[0] = 0.0
    return random_action

def create_optimal_policy(controller):
    """
    Factory function to create a policy function using an OptimalController instance.
    
    Note: dm_control's acrobot only accepts 1 action (elbow torque).
    The controller computes 2 controls [u1, u2], we use u2 (elbow).
    
    Args:
        controller: OptimalController instance
    
    Returns:
        policy_fn: A function that takes timestep and returns action
    """
    def optimal_policy(timestep):
        """Use TVLQR controller for optimal swingup (elbow control only)."""
        obs = timestep.observation
        # dm_control provides: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2]
        cos1, sin1, cos2, sin2 = obs["orientations"]
        dq1, dq2 = obs["velocity"]
        
        # Reconstruct angles from cos/sin
        q1 = np.arctan2(sin1, cos1)
        q2 = np.arctan2(sin2, cos2)
        
        x_current = np.array([q1, q2, dq1, dq2])
        
        # Get control from optimal controller (returns [u1, u2])
        u = controller.get_control(x_current)

        # If environment provides two actuators, return both, otherwise provide elbow only
        try:
            return np.clip(u, action_spec.minimum, action_spec.maximum)
        except Exception:
            # Fallback: return elbow only
            u_elbow = np.array([u[1]])
            return np.clip(u_elbow, action_spec.minimum, action_spec.maximum)
    
    return optimal_policy



# ─────────────────────────────────────────────
# Collect one episode — returns trajectory dict
# ─────────────────────────────────────────────
def collect_episode(policy_fn, max_steps=500, reset_controller=True):
    """
    Collect one episode of trajectories.
    
    Args:
        policy_fn: Policy function (callable)
        max_steps: Maximum steps per episode
        reset_controller: If True, reset controller state before collecting
    
    Returns:
        Dict with states, actions, rewards
    """
    # Reset controller if it's the OptimalController type
    if reset_controller and hasattr(policy_fn, '__self__'):
        if isinstance(policy_fn.__self__, OptimalController):
            policy_fn.__self__.reset()
    
    timestep = env.reset()
    states, actions, rewards = [], [], []

    while not timestep.last():
        state  = flatten_obs(timestep)
        action = policy_fn(timestep)
        print(f"Step {len(states):3d}")
        states.append(state)
        actions.append(action.copy())

        timestep = env.step(action)
        rewards.append(timestep.reward or 0.0)

        if len(states) >= max_steps:
            break

    return {
        "states":  np.array(states),    # (T, 6)
        "actions": np.array(actions),   # (T, 2) or (T, 1)
        "rewards": np.array(rewards),   # (T,)
    }


# ─────────────────────────────────────────────
# Quick sanity-check: run one episode headlessly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MuJoCo Acrobot with Optimal TVLQR Controller")
    print("=" * 60)
    
    # Initialize the optimal controller
    print("\nInitializing OptimalController...")
    print(f"  x_ref shape: {x_ref.shape}")
    print(f"  u_ref shape: {u_ref.shape}")
    print(f"  K_matrix shape: {K_matrix.shape}")
    
    controller = OptimalController(
        x_ref=x_ref,
        u_ref=u_ref,
        K_matrix=K_matrix,
        max_torque=max_torque,
        dt=0.05
    )
    
    # Create policy function from controller
    optimal_policy = create_optimal_policy(controller)
    
    chosen_policy = optimal_policy
    # chosen_policy = random_policy
    # Test with optimal policy
    print("\n--- Testing Optimal TVLQR Controller ---")
    ep_optimal = collect_episode(chosen_policy, max_steps=10000)
    print(f"  Steps   : {len(ep_optimal['states'])}")
    print(f"  State   : {ep_optimal['states'].shape}   — [cos1,sin1,cos2,sin2,dq1,dq2]")
    print(f"  Action  : {ep_optimal['actions'].shape}  — [u1, u2]")
    print(f"  Return  : {ep_optimal['rewards'].sum():.3f}")
    

    launch_viewer = os.environ.get("ENABLE_DM_CONTROL_VIEWER", "0") == "1" # Optional: launch interactive viewer if env var is set
    render_backend = os.environ.get("MUJOCO_GL", "").lower() # Check if using headless EGL rendering

    if launch_viewer:
        if render_backend == "egl":
            print(
                "\nSkipping viewer: MUJOCO_GL=egl is headless/offscreen and "
                "not compatible with dm_control.viewer.\n"
                "To use the interactive viewer, run with MUJOCO_GL=glfw "
                "and ENABLE_DM_CONTROL_VIEWER=1."
            )
        else:
            from dm_control import viewer
            viewer.launch(env, policy=chosen_policy)
