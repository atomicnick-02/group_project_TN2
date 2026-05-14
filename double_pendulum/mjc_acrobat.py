import numpy as np
import os
from dm_control import suite
import types


# Fixed Variables
GRAVITY = 9.81  # gravity in m/s^2
mass = [1,1]
length = [0.05, 0.049]

l1, l2 = length
m1, m2 = mass
b1, b2 = damping
mu1, mu2 = coulomb_fric

PEAK_TORQUE = 0.04  # peak torque in Nm
torque_limits = jnp.array([-PEAK_TORQUE, PEAK_TORQUE])  # torque limits in Nm


Q = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))
Qfin =  Q
R = jnp.diag(jnp.array([1, 1])) * 0.01
max_torque = 0.07      # Physical Hard Stop
# Initialize x_traj with random positions, but keep start and end fixed
# key = jax.random.PRNGKey(42)
key = jax.random.PRNGKey(0)


path_prefix = os.getcwd() + "/temp_images/"
# read tbe optimal trajectory from file
x_ref = np.loadtxt("trajectory.csv", delimiter=",", skiprows=1).T
u_ref = np.loadtxt("inputs.csv", delimiter=",", skiprows=1).T

# ─────────────────────────────────────────────
# Load environment
# ─────────────────────────────────────────────
env = suite.load(
    domain_name="acrobot",
    task_name="swingup",
    task_kwargs={"time_limit": 10.0},
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

# ─────────────────────────────────────────────
# Helper: flatten observation dict → numpy vector
# ─────────────────────────────────────────────
def flatten_obs(timestep):
    """
    dm_control obs is a dict. Flatten to a single vector.
    For acrobot/swingup:
      - orientations: [cos(θ1), sin(θ1), cos(θ2), sin(θ2)]  shape (4,)
      - velocity:     [θ̇1, θ̇2]                               shape (2,)
    Total: 6-dim state vector.
    """
    return np.concatenate([v.flatten() for v in timestep.observation.values()])


# ─────────────────────────────────────────────
# Energy-based swingup expert controller
# (Furuta / Åström–Furuta style for acrobot)
# This gives you a decent expert to collect demos from.
# ─────────────────────────────────────────────
def expert_policy(timestep):
   
    obs = timestep.observation
    # orientations: [cos1, sin1, cos2, sin2]
    cos1, sin1, cos2, sin2 = obs["orientations"]
    dq1, dq2 = obs["velocity"]
    Kp = 10.0e6  # Proportional gain for balance controller
    Ki = 0.0  # Integral gain for balance controller
    # Compute the total energy of the system
    m1, m2 = 1.0, 1.0  # Mass
    l1, l2 = 1.0, 1.0  # Link lengths
    g = 9.81           # Gravity
    # KP: swingup control based on energy difference
    E = -m1 * g * l1 * cos1 - m2 * g * (l1 * cos1 + l2 * cos2) + 0.5 * m1 * (l1 * dq1)**2 + 0.5 * m2 * ((l1 * dq1)**2 + (l2 * dq2)**2 + 2 * l1 * l2 * dq1 * dq2 * cos2)
    E_desired = -m1 * g * l1 - m2 * g * (l1 + l2)  # Desired energy at the upright position
    energy_error = E - E_desired
    torque = Kp * energy_error * np.sign(sin1 * cos2 - cos1 * sin2) + Ki * energy_error  # Add integral term for better balance
    return np.clip([torque], action_spec.minimum, action_spec.maximum)


# ─────────────────────────────────────────────
# Random policy (baseline / exploration)
# ─────────────────────────────────────────────
def random_policy(timestep):
    return np.random.uniform(
        action_spec.minimum,
        action_spec.maximum,
        action_spec.shape,
    )


# ─────────────────────────────────────────────
# Collect one episode — returns trajectory dict
# ─────────────────────────────────────────────
def collect_episode(policy_fn, max_steps=500):
    timestep = env.reset()
    states, actions, rewards = [], [], []

    while not timestep.last():
        state  = flatten_obs(timestep)
        action = policy_fn(timestep)

        states.append(state)
        actions.append(action.copy())

        timestep = env.step(action)
        rewards.append(timestep.reward or 0.0)

        if len(states) >= max_steps:
            break

    return {
        "states":  np.array(states),    # (T, 6)
        "actions": np.array(actions),   # (T, 1)
        "rewards": np.array(rewards),   # (T,)
    }


# ─────────────────────────────────────────────
# Quick sanity-check: run one episode headlessly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nCollecting 1 expert episode (headless)...")
    ep = collect_episode(expert_policy, max_steps=500)
    print(f"  Steps   : {len(ep['states'])}")
    print(f"  State   : {ep['states'].shape}   — [cos1,sin1,cos2,sin2,dq1,dq2]")
    print(f"  Action  : {ep['actions'].shape}  — torque on joint 2")
    print(f"  Return  : {ep['rewards'].sum():.3f}")

   
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
            viewer.launch(env, policy=expert_policy)
