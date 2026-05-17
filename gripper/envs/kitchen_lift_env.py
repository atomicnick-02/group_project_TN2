from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
import numpy as np

from main_dir.utils.xml_wraps import YCBObject


class KitchenLiftEnv(ManipulationEnv):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
	# ------------------------------------------------------------------
	# Model setup
	# ------------------------------------------------------------------

	def _load_model(self):
		super()._load_model()

		mujoco_arena = TableArena(
			table_full_size=(0.8, 0.8, 0.05),
			table_friction=(5.0, 1e-2, 1e-3),
		)
		mujoco_arena.set_origin([0, 0, 0])
		self.robots[0].robot_model.set_base_xpos([-0.5, 0, 0.0])
		object_names = [
			"h_cups",
			"a_cups",
			"peach",
			"plate",
			"lemon",
			"bowl",
			"lemon",
			# Add more object names here if needed
		]
		# Create YCBObject instances; allow duplicate base names by assigning unique
		# instance names (e.g., "lemon", "lemon_1", "lemon_2").
		name_counts = {}
		ycb_objects = []
		for name in object_names:
			count = name_counts.get(name, 0)
			instance_name = name if count == 0 else f"{name}_{count}"
			name_counts[name] = count + 1
			ycb_objects.append(
				YCBObject(
					object_name=name,
					instance_name=instance_name,
					ycb_root="main_dir/assets/ycb/",
				)
			)
		self.ycb_objects = ycb_objects
		
		self.placement_initializer = UniformRandomSampler(
			name="ObjectSampler",
			mujoco_objects=self.ycb_objects,
			ensure_valid_placement=True,
			ensure_object_boundary_in_range=True,
			x_range=[-0.4, 0.4],
			y_range=[-0.4, 0.4],
			reference_pos=(0, 0, 0.8),
			z_offset=0.01,
			rotation=None,
			rotation_axis="z",
		)
		self.model = ManipulationTask(
			mujoco_arena=mujoco_arena,
			mujoco_robots=[robot.robot_model for robot in self.robots],
			mujoco_objects=self.ycb_objects,
		)

	def _setup_references(self):
		"""
		Set up references in the simulation.
		
		Calls parent setup.
		"""
		super()._setup_references()

	# ------------------------------------------------------------------
	# Observations
	# ------------------------------------------------------------------
	def _get_observation(self):
		"""Returns the current observation."""
		obs = super()._get_observation()
		return obs
	# ------------------------------------------------------------------
	# Reset
	# ------------------------------------------------------------------

	def _reset_internal(self):
		super()._reset_internal()

		object_placements = self.placement_initializer.sample()
		# Place each YCB object returned in self.ycb_objects
		for obj in self.ycb_objects:
			pos, quat, _ = object_placements[obj.name]
			# Support batched samples
			if hasattr(pos, "ndim") and pos.ndim > 1:
				pos = pos[0]
			if hasattr(quat, "ndim") and quat.ndim > 1:
				quat = quat[0]
			self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, quat]))

	# ------------------------------------------------------------------
	# Reward
	# ------------------------------------------------------------------

	def reward(self, action=None):
		return 0.0