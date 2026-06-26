import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

current_dir = Path(__file__).resolve().parent


class DoublePendulumEnv(gym.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

	def __init__(self, render_mode=None, frame_skip=10):
		super().__init__()
		self.model = mujoco.MjModel.from_xml_path(str(current_dir / "dp.xml"))
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
		# Optional custom initial state via options={"qpos": [...], "qvel": [...]};
		# defaults to the hanging-down rest state when omitted (unchanged behavior).
		options = options or {}
		self.data.qpos[:2] = options.get("qpos", [0.0, 0.0])
		self.data.qvel[:2] = options.get("qvel", [0.0, 0.0])
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
	import time

	env = DoublePendulumEnv(render_mode="human")
	obs, _ = env.reset()

	for _ in range(500):
		action = env.action_space.sample()
		obs, reward, terminated, truncated, _ = env.step(action)
		# time.sleep(env.dt)
		angles = obs[:2]
		print(f"shoulder: {angles[0]:.2f}, elbow: {angles[1]:.2f}, reward: {reward:.2f}")
		time.sleep(0.01)  # add a small delay to make the simulation visible   
		if terminated or truncated:
			obs, _ = env.reset()

	print("Simulation finished.")
	env.close()
	env.reset()
	# import matplotlib.pyplot as plt
	# env = DoublePendulumEnv(render_mode="rgb_array")
	# env.reset()
	# env.data.qpos[:] = [0.5, -0.3]   # bend it so you can see both links
	# mujoco.mj_forward(env.model, env.data)
	# plt.imsave("frame.png", env.render())