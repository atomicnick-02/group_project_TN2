import os
import sys
import time
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gripper.envs.kitchen_lift_env import MyCustomLiftEnv


def make_ik_controller_config():
    controller_config = load_composite_controller_config(
        controller="BASIC",
        robot="Panda",
    )

    # For a single Panda arm in robosuite v1.5-style composite controllers
    right_arm_cfg = controller_config["body_parts"]["arms"]["right"]

    right_arm_cfg["type"] = "IK_POSE"
    right_arm_cfg["control_delta"] = True

    # Maximum IK delta per policy action
    right_arm_cfg["ik_pos_limit"] = 0.05
    right_arm_cfg["ik_ori_limit"] = 0.25
    right_arm_cfg["converge_steps"] = 5

    return controller_config


def get_cube_position_by_color(env, color):
    """
    Example:
        color = "red"
        searches for cube_red
    """
    target_name = f"cube_{color}"

    for cube, body_id in zip(env.cubes, env.cube_body_ids):
        if cube.name == target_name:
            pos = env.sim.data.body_xpos[body_id].copy()
            return pos, cube.name

    available = [cube.name for cube in env.cubes]
    raise ValueError(f"Could not find {target_name}. Available cubes: {available}")


def make_cartesian_action(obs, target_pos, gripper_cmd, max_step=0.03):
    """
    Creates a relative IK action that moves the end-effector toward target_pos.

    gripper_cmd:
        -1.0 usually means open
        +1.0 usually means close
    If your gripper acts reversed, swap the signs.
    """

    eef_pos = obs["robot0_eef_pos"].copy()

    error = target_pos - eef_pos

    # Limit the Cartesian step to avoid unstable jumps
    dpos = np.clip(error, -max_step, max_step)

    # Keep current orientation for now
    drot = np.zeros(3)

    arm_action = np.concatenate([dpos, drot])

    action_dict = {
        "right": arm_action,
        "right_gripper": np.array([gripper_cmd]),
    }

    return action_dict, np.linalg.norm(error)


def go_to_position(env, robot, obs, target_pos, gripper_cmd, tol=0.015, max_steps=150):
    """
    Move end-effector toward a Cartesian target using IK_POSE.
    """

    for _ in range(max_steps):
        action_dict, error_norm = make_cartesian_action(
            obs=obs,
            target_pos=target_pos,
            gripper_cmd=gripper_cmd,
        )

        action = robot.create_action_vector(action_dict)
        obs, reward, done, info = env.step(action)
        env.render()

        if error_norm < tol:
            break

    return obs


def hold_gripper(env, robot, obs, gripper_cmd, steps=30):
    """
    Keep arm mostly still while opening/closing gripper.
    """

    for _ in range(steps):
        action_dict = {
            "right": np.zeros(6),
            "right_gripper": np.array([gripper_cmd]),
        }

        action = robot.create_action_vector(action_dict)
        obs, reward, done, info = env.step(action)
        env.render()

    return obs


def run_lift_demo(env, target_color="red"):
    obs = env.reset()
    env.render()

    robot = env.robots[0]

    # Useful debug line:
    # It prints expected action keys / dimensions.
    robot.print_action_info()

    cube_pos, cube_name = get_cube_position_by_color(env, target_color)
    print(f"Target object: {cube_name}, position: {cube_pos}")

    # These offsets need tuning for your gripper geometry.
    # cube_pos is the object center, not necessarily the desired gripper center.
    pre_grasp_pos = cube_pos + np.array([0.0, 0.0, 0.12])
    grasp_pos = cube_pos + np.array([0.0, 0.0, 0.035])
    lift_pos = cube_pos + np.array([0.0, 0.0, 0.20])

    # Phase 1: open gripper while moving above object
    obs = go_to_position(
        env=env,
        robot=robot,
        obs=obs,
        target_pos=pre_grasp_pos,
        gripper_cmd=-1.0,
    )

    # Phase 2: descend
    obs = go_to_position(
        env=env,
        robot=robot,
        obs=obs,
        target_pos=grasp_pos,
        gripper_cmd=-1.0,
    )

    # Phase 3: close gripper
    obs = hold_gripper(
        env=env,
        robot=robot,
        obs=obs,
        gripper_cmd=1.0,
        steps=40,
    )

    # Phase 4: lift
    obs = go_to_position(
        env=env,
        robot=robot,
        obs=obs,
        target_pos=lift_pos,
        gripper_cmd=1.0,
        max_steps=200,
    )

    return obs


if __name__ == "__main__":
    controller_config = make_ik_controller_config()

    env = suite.make(
        env_name="MyCustomLiftEnv",
        robots="Panda",
        renderer="mjviewer",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="robot0_eye_in_hand",
        camera_heights=480,
        camera_widths=640,
        ignore_done=True,
        controller_configs=controller_config,
    )

    try:
        run_lift_demo(env, target_color="red")
        time.sleep(1.0)
    finally:
        env.close()