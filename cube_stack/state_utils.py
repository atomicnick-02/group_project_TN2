# state_utils.py
import numpy as np

# ── Stack (2 cubes) ── object vec: 23D, full state: 32D ──────────────────────

OBJ_CUBEA_POS       = slice(0, 3)
OBJ_CUBEA_QUAT      = slice(3, 7)
OBJ_CUBEB_POS       = slice(7, 10)
OBJ_CUBEB_QUAT      = slice(10, 14)
OBJ_GRIPPER_TO_A    = slice(14, 17)
OBJ_GRIPPER_TO_B    = slice(17, 20)
OBJ_CUBEA_TO_CUBEB  = slice(20, 23)

STACK_STATE_DIM = 32


def get_state_from_arrays(object_vec, eef_pos, eef_quat, gripper_qpos):
    """Training path for Stack (2 cubes). object_vec is (23,) from HDF5 obs/object."""
    return np.concatenate([
        object_vec,     # 23 — cubeA/B pos+quat, gripper_to_A/B, A_to_B
        eef_pos,        # 3
        eef_quat,       # 4
        gripper_qpos,   # 2
    ])  # shape: (32,)


def get_state_from_obs(obs_dict):
    """
    Inference path for Stack (2 cubes). Reads robosuite obs dict.
    Produces the identical vector as get_state_from_arrays().
    """
    return np.concatenate([
        obs_dict["cubeA_pos"],              # 3
        obs_dict["cubeA_quat"],             # 4
        obs_dict["cubeB_pos"],              # 3
        obs_dict["cubeB_quat"],             # 4
        obs_dict["gripper_to_cubeA"],       # 3
        obs_dict["gripper_to_cubeB"],       # 3
        obs_dict["cubeA_to_cubeB"],         # 3
        obs_dict["robot0_eef_pos"],         # 3
        obs_dict["robot0_eef_quat"],        # 4
        obs_dict["robot0_gripper_qpos"],    # 2
    ])  # shape: (32,)


# ── StackThree (3 cubes) ── object vec: 39D, full state: 48D ─────────────────
# Layout of the 39D object vector (verified against HDF5 sample values):
#   [0:7]   CubeA pos(3) + quat(4)
#   [7:14]  CubeB pos(3) + quat(4)
#   [14:17] gripper → CubeA
#   [17:20] gripper → CubeB
#   [20:23] CubeA → CubeB
#   [23:30] CubeC pos(3) + quat(4)   ← extra
#   [30:33] gripper → CubeC           ← extra
#   [33:36] CubeA → CubeC             ← extra
#   [36:39] CubeB → CubeC             ← extra

OBJ3_CUBEA_POS       = slice(0, 3)
OBJ3_CUBEA_QUAT      = slice(3, 7)
OBJ3_CUBEB_POS       = slice(7, 10)
OBJ3_CUBEB_QUAT      = slice(10, 14)
OBJ3_GRIPPER_TO_A    = slice(14, 17)
OBJ3_GRIPPER_TO_B    = slice(17, 20)
OBJ3_CUBEA_TO_CUBEB  = slice(20, 23)
OBJ3_CUBEC_POS       = slice(23, 26)
OBJ3_CUBEC_QUAT      = slice(26, 30)
OBJ3_GRIPPER_TO_C    = slice(30, 33)
OBJ3_CUBEA_TO_CUBEC  = slice(33, 36)
OBJ3_CUBEB_TO_CUBEC  = slice(36, 39)

STACK_THREE_STATE_DIM = 48


def get_state_from_arrays_three(object_vec, eef_pos, eef_quat, gripper_qpos):
    """Training path for StackThree (3 cubes). object_vec is (39,) from HDF5 obs/object."""
    return np.concatenate([
        object_vec,     # 39 — cubeA/B/C pos+quat, gripper_to_A/B/C, A_to_B, A_to_C, B_to_C
        eef_pos,        # 3
        eef_quat,       # 4
        gripper_qpos,   # 2
    ])  # shape: (48,)


def get_state_from_obs_three(obs_dict):
    """
    Inference path for StackThree (3 cubes). Reads robosuite obs dict.
    Reconstructs from individual named keys in the same order as the HDF5 obs/object
    layout — the runtime 'object-state' packed key uses a different ordering.
    Produces the identical vector as get_state_from_arrays_three().
    """
    return np.concatenate([
        obs_dict["cubeA_pos"],              # 3  → [0:3]
        obs_dict["cubeA_quat"],             # 4  → [3:7]
        obs_dict["cubeB_pos"],              # 3  → [7:10]
        obs_dict["cubeB_quat"],             # 4  → [10:14]
        obs_dict["gripper_to_cubeA"],       # 3  → [14:17]
        obs_dict["gripper_to_cubeB"],       # 3  → [17:20]
        obs_dict["cubeA_to_cubeB"],         # 3  → [20:23]
        obs_dict["cubeC_pos"],              # 3  → [23:26]
        obs_dict["cubeC_quat"],             # 4  → [26:30]
        obs_dict["gripper_to_cubeC"],       # 3  → [30:33]
        obs_dict["cubeA_to_cubeC"],         # 3  → [33:36]
        obs_dict["cubeB_to_cubeC"],         # 3  → [36:39]
        obs_dict["robot0_eef_pos"],         # 3
        obs_dict["robot0_eef_quat"],        # 4
        obs_dict["robot0_gripper_qpos"],    # 2
    ])  # shape: (48,)


# ── Goal conditioning ────────────────────────────────────────────────────────
# Goal = final-frame cube positions (the stacked configuration), 3 per cube.
STACK_GOAL_DIM       = 6   # cubeA_pos + cubeB_pos
STACK_THREE_GOAL_DIM = 9   # cubeA_pos + cubeB_pos + cubeC_pos

# Combined conditioning dims used by the goal datasets / goal models: base state + goal.
STACK_GOAL_STATE_DIM       = STACK_STATE_DIM       + STACK_GOAL_DIM        # 38
STACK_THREE_GOAL_STATE_DIM = STACK_THREE_STATE_DIM + STACK_THREE_GOAL_DIM  # 57


def get_cube_goal(object_vec):
    """Stack goal: cube positions from a single object vec. shape: (6,)."""
    return np.concatenate([
        object_vec[OBJ_CUBEA_POS],   # 3
        object_vec[OBJ_CUBEB_POS],   # 3
    ])  # shape: (6,)


def get_cube_goal_three(object_vec):
    """StackThree goal: cube positions from a single object vec. shape: (9,)."""
    return np.concatenate([
        object_vec[OBJ3_CUBEA_POS],  # 3
        object_vec[OBJ3_CUBEB_POS],  # 3
        object_vec[OBJ3_CUBEC_POS],  # 3
    ])  # shape: (9,)


# ── Runtime goal synthesis ───────────────────────────────────────────────────
# At inference the cubes are NOT stacked yet, so we cannot read the "final"
# positions like the dataset does. Instead we synthesize the target tower from
# the CURRENT scene: the base cube (cubeB) stays put, upper cubes sit above it.
# The output order must match the dataset goal order (cubeA, cubeB[, cubeC]) so
# train/test conditioning stays consistent.
#
# Stacking order and the per-level z offsets below were measured from the
# stack_three_d0 final frames (mean over 200 demos): order low->high is B -> A -> C,
# cubeA sits ~0.0444 m above cubeB and cubeC ~0.0394 m above cubeA, with the
# stacked cubes xy-aligned to the base within ~1 cm.
STACK_THREE_A_OVER_B = 0.0444  # cubeA height above cubeB (m)
STACK_THREE_C_OVER_A = 0.0394  # cubeC height above cubeA (m)

# 2-cube Stack: not measured here (no stack_d* dataset available); approximate.
CUBE_HEIGHT = 0.04  # meters; center-to-center spacing of stacked cubes


def synthesize_stack_goal(obs_dict, cube_height=CUBE_HEIGHT):
    """Stack goal from current obs: cubeA on top of cubeB. shape: (6,)."""
    base = np.asarray(obs_dict["cubeB_pos"], dtype=np.float32)
    cubeA_goal = base + np.array([0.0, 0.0, cube_height], dtype=np.float32)
    cubeB_goal = base
    return np.concatenate([cubeA_goal, cubeB_goal])  # order: cubeA, cubeB


def synthesize_stack_three_goal(obs_dict):
    """StackThree goal from current obs: B (base) -> A -> C tower. shape: (9,)."""
    base = np.asarray(obs_dict["cubeB_pos"], dtype=np.float32)
    cubeA_goal = base + np.array([0.0, 0.0, STACK_THREE_A_OVER_B], dtype=np.float32)
    cubeC_goal = base + np.array([0.0, 0.0, STACK_THREE_A_OVER_B + STACK_THREE_C_OVER_A], dtype=np.float32)
    cubeB_goal = base
    return np.concatenate([cubeA_goal, cubeB_goal, cubeC_goal])  # order: A, B, C
