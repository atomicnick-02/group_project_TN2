import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

from tqdm import tqdm
from behavioral_cloning import BCPolicyModel
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
H = 8
MODEL_N = 3
MAX_EPISODE_STEPS = 2000
MAX_EPISODES = 50
def denorm_action(a, action_range, action_min): 
	return (a + 1.0) * 0.5 * action_range + action_min

def feat_transform(x, vel_min, vel_range):
	q1 = x[0]
	q2 = x[1]
	v_norm  = 2.0 * (x[2:] - vel_min) / vel_range - 1.0
	feat = np.vstack([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2), v_norm[0], v_norm[1]])
	return feat

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
XML_PATH = repo_root / "double_pendulum" / "dp.xml"

def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class DoublePendulumEnv(gym.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

	def __init__(self, render_mode=None, frame_skip=10):
		super().__init__()
		self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
		self.data = mujoco.MjData(self.model)
		self.frame_skip = frame_skip
		self.dt = self.model.opt.timestep * frame_skip

		# action_space matches the MJCF ctrlrange (Nm)
		tau_max = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
		self.action_space = spaces.Box(low=-tau_max, high=tau_max, dtype=np.float32)

		# obs = [q1, q2, qd1, qd2]
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
		)

		# rendering
		self.render_mode = render_mode
		self._viewer = None       # for "human"
		self._renderer = None     # for "rgb_array"

	def _get_obs(self):
		# wrap angles to [-pi, pi] for better learning performance, but keep velocities as is
		self.data.qpos[0] = (self.data.qpos[0] + np.pi) % (2 * np.pi) - np.pi
		self.data.qpos[1] = (self.data.qpos[1] + np.pi) % (2 * np.pi) - np.pi
		return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		mujoco.mj_resetData(self.model, self.data)
		self.data.qpos[:] = [0.0, 0.0]
		self.data.qvel[:] = [0.0, 0.0]
		mujoco.mj_forward(self.model, self.data)
		return self._get_obs(), {}

	def step(self, action):
		self.data.ctrl[:] = np.clip(action,
									self.action_space.low,
									self.action_space.high)
		for _ in range(self.frame_skip):
			mujoco.mj_step(self.model, self.data)

		obs = self._get_obs()
		goal = np.array([np.pi, 0.0, 0.0, 0.0], dtype=np.float32)
		reward = -float(np.sum((obs - goal) ** 2))
		terminated = False
		truncated = False

		if self.render_mode == "human":
			self.render()

		return obs, reward, terminated, truncated, {}

	def render(self):
		if self.render_mode == "human":
			if self._viewer is None:
				self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
				# set fixed camera once, after the viewer is created
				cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
				self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
				self._viewer.cam.fixedcamid = cam_id
			self._viewer.sync()
			return None

		if self.render_mode == "rgb_array":
			if self._renderer is None:
				self._renderer = mujoco.Renderer(self.model, height=480, width=640)
			self._renderer.update_scene(self.data, camera="side")   # by name
			return self._renderer.render()

	def close(self):
		if self._viewer is not None:
			self._viewer.close()
			self._viewer = None
		if self._renderer is not None:
			self._renderer.close()
			self._renderer = None


if __name__ == "__main__":
	# Normalization stats: prefer the model-specific file, fall back to the generic one.
	stats_path = current_dir / f"norm_stats_{MODEL_N}.json"
	if not stats_path.exists():
		stats_path = current_dir / "norm_stats.json"
	with open(stats_path, 'r') as file:
		data = json.load(file)
	action_min = np.array(data['action_min'])
	action_range = np.array(data['action_max']) - np.array(data['action_min'])
	vel_min = np.array(data['vel_min'])
	vel_range = np.array(data['vel_max']) - np.array(data['vel_min'])

	ckpt_path = current_dir / f"bc_policy_{MODEL_N}.pt"
	model = BCPolicyModel(H)
	model.load_state_dict(torch.load(str(ckpt_path),
												   map_location=lambda storage, loc: storage))
	model.eval()

	angle_tol = 0.2
	max_holding_time = 10
	env = DoublePendulumEnv()  # render_mode="human" to watch

	# Sweep the executed-chunk horizon: replan every `exec_h` steps and execute the
	# first `exec_h` actions of the predicted chunk before querying the policy again.
	exec_horizons = list(range(1, H + 1))          # 1 .. H
	successes = np.zeros(len(exec_horizons))
	for hi, exec_h in enumerate(exec_horizons):
		for episode in tqdm(range(MAX_EPISODES), desc=f"horizon {exec_h}"):
			env.reset()
			# Randomize the initial state, then re-evaluate kinematics so obs is fresh.
			env.data.qpos[:] = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]
			env.data.qvel[:] = [np.random.uniform(-4.0, 4.0), np.random.uniform(-4.0, 4.0)]
			mujoco.mj_forward(env.model, env.data)
			obs = env._get_obs()

			holding_time = 0
			action_chunk = None
			step_in_chunk = exec_h                  # force a replan on the first step
			for _ in range(200):
				if step_in_chunk >= exec_h:
					state = feat_transform(obs, vel_min, vel_range)
					state = torch.from_numpy(state.T).float()
					action_chunk = model(state).squeeze(0).detach().numpy()
					step_in_chunk = 0
				action = denorm_action(action_chunk[step_in_chunk, :], action_range, action_min)
				obs, reward, terminated, truncated, _ = env.step(action)
				step_in_chunk += 1

				angles = obs[:2]
				if abs(wrap_to_pi(angles[0] - np.pi)) + abs(wrap_to_pi(angles[1] - 0)) < angle_tol:
					holding_time += 1
				else:
					holding_time = 0

				if holding_time >= max_holding_time:
					successes[hi] += 1
					break

	print("Simulation finished.")
	env.close()
	successes /= MAX_EPISODES
	labels = [str(h) for h in exec_horizons]

	# Create the bar chart (without plt.figure() as per guidelines)
	plt.bar(labels, successes, color='blue', edgecolor='black')
	plt.xlabel('Horizon Length')
	plt.ylabel('Success rate')
	plt.title('Success rate for different horizon lengths')
	plt.tight_layout()
	plt.savefig(str(current_dir / 'bar_graph.png'))
	plt.show()
	plt.close()
	df = pd.DataFrame({'horizon': labels, 'success_rate': successes})
	df.to_csv(str(current_dir / 'data_pandas.csv'), index=False)
