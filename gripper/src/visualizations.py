import numpy as np
import cv2 
import torch


def vis_bb(image, result, window_name="Grounding DINO Results", scale=0.75):
	img_np = np.array(image)
	img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

	for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
		box_int = [int(i) for i in box.tolist()]
		x_min, y_min, x_max, y_max = box_int

		cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

		text = f"{label} ({score.item():.2f})"
		cv2.putText(
			img_cv,
			text,
			(x_min, max(y_min - 10, 0)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.4,
			(0, 255, 0),
			1
		)

	width = int(img_cv.shape[1] * scale)
	height = int(img_cv.shape[0] * scale)
	img_cv = cv2.resize(img_cv, (width, height), interpolation=cv2.INTER_AREA)

	cv2.imshow(window_name, img_cv)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def vis_seg(
	masks,
	image,
	labels=None,
	boxes=None,
	scores=None,
	alpha=0.6,
	scale=0.75,
	window_name="Segmented Objects Overlay",
	save_path=None,
	show=True
):
	img_np = np.array(image)
	img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

	mask_data = masks[0]

	if isinstance(mask_data, torch.Tensor):
		mask_data = mask_data.cpu().numpy()

	if scores is not None and isinstance(scores, torch.Tensor):
		scores = scores.cpu().numpy()

	if boxes is not None:
		if isinstance(boxes, torch.Tensor):
			boxes = boxes.cpu().numpy()
		boxes = np.array(boxes)

	# Expected shape: (num_objects, num_candidate_masks, H, W)
	# With multimask_output=False, this is usually: (num_objects, 1, H, W)
	if mask_data.ndim != 4:
		raise ValueError(f"Unexpected mask shape: {mask_data.shape}")

	num_objects, num_candidates, H, W = mask_data.shape

	overlay = img_cv.copy()

	colors = [
		(0, 0, 255),
		(0, 255, 0),
		(255, 0, 0),
		(0, 255, 255),
		(255, 0, 255),
		(255, 255, 0),
		(0, 128, 255),
		(128, 0, 255),
		(255, 128, 0),
		(128, 255, 0),
	]

	selected_masks = []

	for i in range(num_objects):
		if scores is not None:
			best_idx = int(np.argmax(scores[0, i]))
		else:
			best_idx = 0

		obj_mask = mask_data[i, best_idx] > 0
		selected_masks.append(obj_mask)

		color = colors[i % len(colors)]
		overlay[obj_mask] = color

	blended = cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0)

	for i, obj_mask in enumerate(selected_masks):
		color = colors[i % len(colors)]

		mask_uint8 = obj_mask.astype(np.uint8) * 255
		contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)

		if labels is not None and i < len(labels):
			label_text = str(labels[i])
		else:
			label_text = f"object_{i + 1}"

		# Prefer label placement from bounding box
		if boxes is not None and i < len(boxes):
			x1, y1, x2, y2 = boxes[i].astype(int)
			x_text = x1
			y_text = max(y1 - 10, 20)
		elif len(contours) > 0:
			largest_contour = max(contours, key=cv2.contourArea)
			x, y, w, h = cv2.boundingRect(largest_contour)
			x_text = x
			y_text = max(y - 10, 20)
		else:
			x_text = 10
			y_text = 30 + i * 25

		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.6
		thickness = 2

		(text_w, text_h), baseline = cv2.getTextSize(
			label_text,
			font,
			font_scale,
			thickness
		)

		rect_x1 = max(x_text, 0)
		rect_y1 = max(y_text - text_h - 6, 0)
		rect_x2 = min(x_text + text_w + 6, blended.shape[1] - 1)
		rect_y2 = min(y_text + baseline + 2, blended.shape[0] - 1)

		cv2.rectangle(
			blended,
			(rect_x1, rect_y1),
			(rect_x2, rect_y2),
			(0, 0, 0),
			-1
		)

		cv2.putText(
			blended,
			label_text,
			(rect_x1 + 3, rect_y2 - baseline - 2),
			font,
			font_scale,
			(255, 255, 255),
			thickness
		)

	width = int(blended.shape[1] * scale)
	height = int(blended.shape[0] * scale)
	blended_resized = cv2.resize(blended, (width, height), interpolation=cv2.INTER_AREA)

	if save_path is not None:
		cv2.imwrite(str(save_path), blended_resized)

	if show:
		cv2.imshow(window_name, blended_resized)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return blended_resized

