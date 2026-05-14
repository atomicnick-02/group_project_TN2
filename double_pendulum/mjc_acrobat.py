import numpy as np
import os
from dm_control import suite

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

    # Upright = cos1 close to -1 (top), cos2 close to 1
    # Energy pumping: apply torque in direction of velocity
    # scaled by how far we are from upright
    height = -(cos1 + cos2)  # max = 2 at top, min = -2 at bottom

    # Switch to LQR-style damping near top
    if height > 1.6:
        # PD balance controller
        theta1 = np.arctan2(sin1, cos1)   # 0 = hanging, ±π = upright
        torque = -2.0 * (theta1 - np.pi) - 0.5 * dq1
    else:
        # Energy pumping
        torque = 2.0 * np.sign(dq2) * (1.0 - height / 2.0)

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
