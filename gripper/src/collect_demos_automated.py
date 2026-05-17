import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.devices import Keyboard
import sys, cv2


from main_dir.envs.kitchen_lift_env import KitchenLiftEnv

# ---------------------------------------------------------
# Controller config
# ---------------------------------------------------------

controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")

print("Controller config keys:", controller_config.keys())
print("Body part keys:", controller_config["body_parts"].keys())

right_arm_cfg = controller_config["body_parts"]["right"]
use_opencv_render = False

env = suite.make(
    env_name="KitchenLiftEnv",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,          # ← enable camera observations
    camera_names=["agentview"],   # ← which cameras to capture
    camera_heights=480,
    camera_widths=640,
    ignore_done=True,
    controller_configs=controller_config,
)
def show_camera(obs):
	if not use_opencv_render:
		return

	frame = obs["robot0_eye_in_hand_image"][::-1]
	frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	cv2.imshow("robot0_eye_in_hand", frame_bgr)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		cv2.destroyAllWindows()
		env.close()
		sys.exit()


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

robot = env.robots[0]
env.reset()
print(20*"-")
print("starting main loop...")
print(20*"-")


device = Keyboard(env=env)
env.viewer.add_keypress_callback(device.on_press)
robot = env.robots[0]
device.start_control()
while True:
    
	while True:
		input_ac_dict = device.input2action()
		if input_ac_dict is None:
			break

		action_dict = input_ac_dict.copy()
		action_dict["right"] = input_ac_dict["right_delta"]
		action_dict["right_gripper"] = input_ac_dict["right_gripper"]
		action = robot.create_action_vector(action_dict)

		obs, reward, done, info = env.step(action)
		env.render()
		# Optional: show camera frame
		# show_camera(obs)

		# time.sleep(1.0)
	print("Resetting environment...")
	print(input_ac_dict)
	obs = env.reset()
	device.start_control()
	
	