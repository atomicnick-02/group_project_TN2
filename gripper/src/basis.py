import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.devices import Keyboard
import time
import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gripper.envs.cubes_lift_env import MyCustomLiftEnv
from gripper.src.visualizations import vis_seg 

controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")

env = suite.make(
	env_name="MyCustomLiftEnv",
	robots="Panda",
	renderer="mjviewer",
	has_renderer=True,
	has_offscreen_renderer=True,
	use_camera_obs=True,          
	camera_names= "robot0_eye_in_hand",  
	camera_heights=480,
	camera_widths=640,
	ignore_done=True,
	controller_configs=controller_config,
	# Custom VLM params passed down
	use_vlm=False,

)
# vailable camera views:
#   [0] frontview
#   [1] birdview
#   [2] agentview
#   [3] sideview
#   [4] robot0_robotview
#   [5] robot0_eye_in_hand

vlm_display_img = None  # Cache for the last generated VLM overlay
use_opencv_render = False  # Flag to allow opecnCV rendering of VLM results
env.reset()

device = Keyboard(env=env)
env.viewer.add_keypress_callback(device.on_press)
robot = env.robots[0]

while True:
	obs = env.reset()
	env.render()
	device.start_control()

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

		# ── Agentview Camera feed display ──
		if use_opencv_render:
			frame = obs["robot0_eye_in_hand_image"]  # (H, W, C) in RGB format  
			frame_bgr = cv2.cvtColor(frame[::-1], cv2.COLOR_RGB2BGR)
			cv2.imshow("agentview", frame_bgr)

		# ── VLM Eye-in-Hand display logic ──
		eye_frame = obs["robot0_eye_in_hand_image"][::-1] # Correct orientation
		
		if info.get("vlm_updated", False) and info.get("vlm") is not None:
			vlm_res = info["vlm"]
			
			if vlm_res.get("masks") is not None and len(vlm_res["masks"]) > 0:
				# vis_seg strictly expects masks[0] format (N, num_candidates, H, W)
				masks_4d = np.expand_dims(vlm_res["masks"], axis=1) 
				
				# Turn OFF 'show' and 'save_path' to meet your requirements
				vlm_display_img = vis_seg(
					masks=[masks_4d],
					image=eye_frame,
					labels=vlm_res["labels"],
					boxes=vlm_res["boxes"],
					show=False,
					scores=None,
					save_path=None    
				)
			else:
				vlm_display_img = cv2.cvtColor(eye_frame, cv2.COLOR_RGB2BGR)

		# Keep showing the cached VLM frame, or live camera if no VLM processed yet
		if vlm_display_img is not None and use_opencv_render:
			cv2.imshow("VLM Pipeline (Eye in Hand)", vlm_display_img)
		elif use_opencv_render:
			cv2.imshow("VLM Pipeline (Eye in Hand)", cv2.cvtColor(eye_frame, cv2.COLOR_RGB2BGR))

		if cv2.waitKey(1) & 0xFF == ord("q") and use_opencv_render:
			cv2.destroyAllWindows()
			env.close()
			sys.exit()

	cv2.destroyAllWindows()
	env.close()