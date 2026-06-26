import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from state_utils import (
    get_state_from_arrays, STACK_STATE_DIM, STACK_GOAL_DIM,
    get_state_from_arrays_three, STACK_THREE_STATE_DIM, STACK_THREE_GOAL_DIM,
    get_cube_goal, get_cube_goal_three,
)


class ShortHorizonRoboticsDataset(Dataset):
    """Dataset for the Stack task (2 cubes). State dim: 32."""

    def __init__(self, file_path, horizon=8):
        self.horizon = horizon
        self.inputs  = []
        self.targets = []

        with h5py.File(file_path, "r") as f:
            data_group = f["data"]

            for demo_key in data_group.keys():
                demo = data_group[demo_key]

                actions      = demo["actions"][:]                  # (T, 7)
                object_vecs  = demo["obs/object"][:]               # (T, 23)
                eef_pos      = demo["obs/robot0_eef_pos"][:]       # (T, 3)
                eef_quat     = demo["obs/robot0_eef_quat"][:]      # (T, 4)
                gripper_qpos = demo["obs/robot0_gripper_qpos"][:]  # (T, 2)

                T = actions.shape[0]

                for t in range(T - horizon + 1):
                    state_t = get_state_from_arrays(
                        object_vec   = object_vecs[t],
                        eef_pos      = eef_pos[t],
                        eef_quat     = eef_quat[t],
                        gripper_qpos = gripper_qpos[t],
                    )  # (32,)

                    self.inputs.append(state_t)
                    self.targets.append(actions[t: t + horizon])

        self.inputs  = torch.tensor(np.array(self.inputs),  dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

        assert self.inputs.shape[1] == STACK_STATE_DIM, \
            f"State dim mismatch: got {self.inputs.shape[1]}, expected {STACK_STATE_DIM}"
        assert self.targets.shape[1:] == (horizon, 7), \
            f"Action shape mismatch: got {self.targets.shape[1:]}"

        print(f"Dataset loaded:  {len(self.inputs)} samples")
        print(f"Input shape:     {self.inputs.shape}")   # (N, 32)
        print(f"Target shape:    {self.targets.shape}")  # (N, 8, 7)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ShortHorizonRoboticsDatasetGoal(Dataset):
    """
    Stack task (2 cubes) with a goal appended to the conditioning vector. The goal
    is the final-frame cube positions (the stacked configuration), so each window's
    state is:
        [base_state (32) | goal (6)] -> 38 dims.
    Returns (state, action_seq), a drop-in for the existing training pipeline.
    """

    def __init__(self, file_path, horizon=8):
        self.horizon = horizon
        self.inputs  = []
        self.targets = []

        with h5py.File(file_path, "r") as f:
            data_group = f["data"]

            for demo_key in data_group.keys():
                demo = data_group[demo_key]

                actions        = demo["actions"][:]                     # (T, 7)
                object_vecs    = demo["obs/object"][:]                  # (T, 23)
                eef_pos        = demo["obs/robot0_eef_pos"][:]          # (T, 3)
                eef_quat       = demo["obs/robot0_eef_quat"][:]         # (T, 4)
                gripper_qpos   = demo["obs/robot0_gripper_qpos"][:]     # (T, 2)

                T = actions.shape[0]

                # Goal: cube positions in the final frame (stacked configuration).
                goal = get_cube_goal(object_vecs[-1])  # (6,)

                for t in range(T - horizon + 1):
                    state_t = get_state_from_arrays(
                        object_vec   = object_vecs[t],
                        eef_pos      = eef_pos[t],
                        eef_quat     = eef_quat[t],
                        gripper_qpos = gripper_qpos[t],
                    )  # (32,)
                    state_t = np.concatenate([state_t, goal])  # (38,)

                    self.inputs.append(state_t)
                    self.targets.append(actions[t: t + horizon])

        self.inputs  = torch.tensor(np.array(self.inputs),  dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

        assert self.inputs.shape[1] == STACK_STATE_DIM + STACK_GOAL_DIM, \
            f"State dim mismatch: got {self.inputs.shape[1]}, expected {STACK_STATE_DIM + STACK_GOAL_DIM}"
        assert self.targets.shape[1:] == (horizon, 7), \
            f"Action shape mismatch: got {self.targets.shape[1:]}"

        print(f"Dataset loaded:  {len(self.inputs)} samples")
        print(f"Input shape:     {self.inputs.shape}")   # (N, 38)
        print(f"Target shape:    {self.targets.shape}")  # (N, 8, 7)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ShortHorizonRoboticsDatasetThree(Dataset):
    """Dataset for the StackThree task (3 cubes). State dim: 48."""

    def __init__(self, file_path, horizon=8):
        self.horizon = horizon
        self.inputs  = []
        self.targets = []

        with h5py.File(file_path, "r") as f:
            data_group = f["data"]

            for demo_key in data_group.keys():
                demo = data_group[demo_key]

                actions      = demo["actions"][:]                  # (T, 7)
                object_vecs  = demo["obs/object"][:]               # (T, 39)
                eef_pos      = demo["obs/robot0_eef_pos"][:]       # (T, 3)
                eef_quat     = demo["obs/robot0_eef_quat"][:]      # (T, 4)
                gripper_qpos = demo["obs/robot0_gripper_qpos"][:]  # (T, 2)

                assert object_vecs.shape[1] == 39, \
                    f"Expected 39D object vec for StackThree, got {object_vecs.shape[1]}D. Wrong dataset?"

                T = actions.shape[0]

                for t in range(T - horizon + 1):
                    state_t = get_state_from_arrays_three(
                        object_vec   = object_vecs[t],
                        eef_pos      = eef_pos[t],
                        eef_quat     = eef_quat[t],
                        gripper_qpos = gripper_qpos[t],
                    )  # (48,)

                    self.inputs.append(state_t)
                    self.targets.append(actions[t: t + horizon])

        self.inputs  = torch.tensor(np.array(self.inputs),  dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

        assert self.inputs.shape[1] == STACK_THREE_STATE_DIM, \
            f"State dim mismatch: got {self.inputs.shape[1]}, expected {STACK_THREE_STATE_DIM}"
        assert self.targets.shape[1:] == (horizon, 7), \
            f"Action shape mismatch: got {self.targets.shape[1:]}"

        print(f"Dataset loaded:  {len(self.inputs)} samples")
        print(f"Input shape:     {self.inputs.shape}")   # (N, 48)
        print(f"Target shape:    {self.targets.shape}")  # (N, 8, 7)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ShortHorizonRoboticsDatasetThreeGoal(Dataset):
    """
    StackThree task (3 cubes) with a goal appended to the conditioning vector. The
    goal is the final-frame cube positions (the stacked configuration), so each
    window's state is:
        [base_state (48) | goal (9)] -> 57 dims.
    Returns (state, action_seq), a drop-in for the existing training pipeline.
    """

    def __init__(self, file_path, horizon=8):
        self.horizon = horizon
        self.inputs  = []
        self.targets = []

        with h5py.File(file_path, "r") as f:
            data_group = f["data"]

            for demo_key in data_group.keys():
                demo = data_group[demo_key]

                actions        = demo["actions"][:]                     # (T, 7)
                object_vecs    = demo["obs/object"][:]                  # (T, 39)
                eef_pos        = demo["obs/robot0_eef_pos"][:]          # (T, 3)
                eef_quat       = demo["obs/robot0_eef_quat"][:]         # (T, 4)
                gripper_qpos   = demo["obs/robot0_gripper_qpos"][:]     # (T, 2)

                assert object_vecs.shape[1] == 39, \
                    f"Expected 39D object vec for StackThree, got {object_vecs.shape[1]}D. Wrong dataset?"

                T = actions.shape[0]

                # Goal: cube positions in the final frame (stacked configuration).
                goal = get_cube_goal_three(object_vecs[-1])  # (9,)

                for t in range(T - horizon + 1):
                    state_t = get_state_from_arrays_three(
                        object_vec   = object_vecs[t],
                        eef_pos      = eef_pos[t],
                        eef_quat     = eef_quat[t],
                        gripper_qpos = gripper_qpos[t],
                    )  # (48,)
                    state_t = np.concatenate([state_t, goal])  # (57,)

                    self.inputs.append(state_t)
                    self.targets.append(actions[t: t + horizon])

        self.inputs  = torch.tensor(np.array(self.inputs),  dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

        assert self.inputs.shape[1] == STACK_THREE_STATE_DIM + STACK_THREE_GOAL_DIM, \
            f"State dim mismatch: got {self.inputs.shape[1]}, expected {STACK_THREE_STATE_DIM + STACK_THREE_GOAL_DIM}"
        assert self.targets.shape[1:] == (horizon, 7), \
            f"Action shape mismatch: got {self.targets.shape[1:]}"

        print(f"Dataset loaded:  {len(self.inputs)} samples")
        print(f"Input shape:     {self.inputs.shape}")   # (N, 57)
        print(f"Target shape:    {self.targets.shape}")  # (N, 8, 7)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
