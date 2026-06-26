import numpy as np

# Only record gripper-cube alignment when the gripper is within this distance of the cube
APPROACH_RADIUS = 0.10  # metres


def _quat_rotate_vec(quat, vec):
    """Rotate vec by quaternion in (x, y, z, w) robosuite convention."""
    x, y, z, w = quat
    u = np.array([x, y, z])
    return (2.0 * np.dot(u, vec) * u
            + (w * w - np.dot(u, u)) * vec
            + 2.0 * w * np.cross(u, vec))


class TrajectoryMetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal accumulators between evaluation episodes."""
        self.episode_jitters = []
        self.episode_path_lengths = []
        self.episode_joint_costs = []
        self.episode_grasp_alignments = []
        self.episode_mses = []

    def update_step_metrics(self, predicted_horizon, true_horizon=None):
        """
        Call once per planning step (every time the model generates an 8-step plan).

        predicted_horizon: numpy array of shape (horizon, action_dim) -> e.g., (8, 7)
        true_horizon: optional numpy array of shape (horizon, action_dim) from expert data
        """
        # 1. Jitter — mean squared difference between consecutive steps
        action_diffs = np.diff(predicted_horizon, axis=0)  # (horizon-1, action_dim)
        self.episode_jitters.append(np.mean(action_diffs ** 2))

        # 2. Path effort — total absolute xyz translation commands
        self.episode_path_lengths.append(np.sum(np.abs(predicted_horizon[:, :3])))

        # 3. Joint cost — total absolute actuation across all 7 action dimensions
        self.episode_joint_costs.append(np.sum(np.abs(predicted_horizon)))

        # 4. Ground-truth MSE (optional)
        if true_horizon is not None:
            self.episode_mses.append(np.mean((predicted_horizon - true_horizon) ** 2))

    def update_grasp_alignment(self, eef_quat, gripper_to_cube_vec):
        """
        Call once per environment step to track how well the gripper is oriented
        toward the target cube during close approach.

        eef_quat          : (4,) end-effector quaternion in (x, y, z, w) convention
        gripper_to_cube_vec: (3,) vector pointing from the gripper to the target cube

        The angle between the gripper z-axis (approach direction) and the direction
        to the cube is recorded only when the gripper is within APPROACH_RADIUS metres
        of the cube. 0° = gripper pointing directly at the cube (ideal alignment).
        """
        dist = np.linalg.norm(gripper_to_cube_vec)
        if dist > APPROACH_RADIUS or dist < 1e-6:
            return

        gripper_z_world = _quat_rotate_vec(eef_quat, np.array([0.0, 0.0, 1.0]))
        cube_dir = gripper_to_cube_vec / dist
        cos_angle = np.clip(np.dot(gripper_z_world, cube_dir), -1.0, 1.0)
        # abs: gripper z-axis might point toward or away from cube depending on wrist pose
        angle_deg = np.degrees(np.arccos(np.abs(cos_angle)))
        self.episode_grasp_alignments.append(angle_deg)

    def get_episode_summary(self):
        """Computes statistical averages for the current completed episode rollout."""
        summary = {
            "mean_jitter":       np.mean(self.episode_jitters)       if self.episode_jitters       else 0.0,
            "total_path_effort": np.sum (self.episode_path_lengths)  if self.episode_path_lengths  else 0.0,
            "total_joint_cost":  np.sum (self.episode_joint_costs)   if self.episode_joint_costs   else 0.0,
        }
        if self.episode_grasp_alignments:
            summary["mean_approach_alignment_deg"] = np.mean(self.episode_grasp_alignments)
            summary["best_approach_alignment_deg"] = np.min (self.episode_grasp_alignments)
        if self.episode_mses:
            summary["mean_ground_truth_mse"] = np.mean(self.episode_mses)
        return summary
