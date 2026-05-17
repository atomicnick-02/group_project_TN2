from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
import numpy as np
import time
from gripper.src.perception import VLMPerception

# Map RGBA -> color name used as VLM text label
COLOR_PALETTE = {
	"red":    [1.0, 0.0, 0.0, 1.0],
	"green":  [0.0, 1.0, 0.0, 1.0],
	"blue":   [0.0, 0.0, 1.0, 1.0],
	"yellow": [1.0, 1.0, 0.0, 1.0],
	"orange": [1.0, 0.5, 0.0, 1.0],
	"purple": [0.5, 0.0, 1.0, 1.0],
}

NUM_CUBES = 4

class MyCustomLiftEnv(ManipulationEnv):

	def __init__(self, use_vlm=True, vlm_camera="robot0_eye_in_hand", vlm_interval=3.0, **kwargs):
		self.use_vlm = use_vlm
		self.vlm_camera = vlm_camera
		self.vlm = VLMPerception() if use_vlm else None
		self.cube_colors = []   
		self.vlm_result = None  
		
		# New: Track time to prevent freezing the simulation
		self.vlm_interval = vlm_interval
		self.last_vlm_time = time.time()
		
		super().__init__(**kwargs)
	# ------------------------------------------------------------------
	# Model setup
	# ------------------------------------------------------------------

	def _load_model(self):
		super()._load_model()

		mujoco_arena = TableArena(
			table_full_size=(0.8, 0.8, 0.05),
			table_friction=(1.0, 5e-3, 1e-4),
		)
		mujoco_arena.set_origin([0, 0, 0])
		self.robots[0].robot_model.set_base_xpos([-0.5, 0, 0.0])

		# Pick NUM_CUBES distinct colors randomly
		all_color_names = list(COLOR_PALETTE.keys())
		chosen_names = np.random.choice(all_color_names, size=NUM_CUBES, replace=False).tolist()
		self.cube_colors = chosen_names  # e.g. ["red", "blue", "yellow", "green"]

		self.cubes = [
			BoxObject(
				name=f"cube_{color}",
				size=[0.025, 0.025, 0.025],
				rgba=COLOR_PALETTE[color],
			)
			for color in self.cube_colors
		]

		self.placement_initializer = UniformRandomSampler(
			name="ObjectSampler",
			mujoco_objects=self.cubes,
			ensure_valid_placement=True,
			ensure_object_boundary_in_range=True,
			x_range=[-0.3, 0.3],
			y_range=[-0.3, 0.3],
			reference_pos=(0, 0, 0.8),
			z_offset=0.025,
			rotation=None,
			rotation_axis='z',
		)

		self.model = ManipulationTask(
			mujoco_arena=mujoco_arena,
			mujoco_robots=[robot.robot_model for robot in self.robots],
			mujoco_objects=self.cubes,
		)

	def _setup_references(self):
		super()._setup_references()
		self.cube_body_ids = [
			self.sim.model.body_name2id(cube.root_body) for cube in self.cubes
		]

	# ------------------------------------------------------------------
	# Reset
	# ------------------------------------------------------------------

	def _reset_internal(self):
		super()._reset_internal()

		object_placements = self.placement_initializer.sample()

		for obj_name, obj_data in object_placements.items():
			pos, quat, _ = obj_data
			for cube in self.cubes:
				if obj_name == cube.name:
					self.sim.data.set_joint_qpos(
						cube.joints[0], np.concatenate([pos, quat])
					)

	# ------------------------------------------------------------------
	# Step — hook VLM after physics step
	# ------------------------------------------------------------------

	def step(self, action):
		obs, reward, done, info = super().step(action)
		info["vlm_updated"] = False

		if self.use_vlm:
			current_time = time.time()
			# Only trigger VLM if the interval has passed
			if current_time - self.last_vlm_time >= self.vlm_interval:
				self.vlm_result = self._run_vlm(obs)
				self.last_vlm_time = current_time
				info["vlm_updated"] = True
			
			info["vlm"] = self.vlm_result

		return obs, reward, done, info
	# ------------------------------------------------------------------
	# VLM
	# ------------------------------------------------------------------

	def _run_vlm(self, obs: dict) -> dict:
		cam_key = f"{self.vlm_camera}_image"
		if cam_key not in obs:
			return {}

		# Correct the image orientation. Robosuite renders upside-down by default.
		image_rgb = obs[cam_key][::-1]  
		text_labels = [f"{color} cube" for color in self.cube_colors]
		mask_height = 100 
		image_rgb[-mask_height:, :, :] = 0
		result = self.vlm.predict(image_rgb=image_rgb, text_labels=text_labels)
		return result
	# ------------------------------------------------------------------
	# Reward
	# ------------------------------------------------------------------

	def reward(self, action=None):
		return 0.0