import time
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers import Sam2Processor, Sam2Model


# ============================================================
# Settings
# ============================================================

CAMERA_INDEX = 0

TARGET_FPS = 2.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DETECTION_MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
SEGMENTATION_MODEL_NAME = "facebook/sam2.1-hiera-tiny"

TEXT_LABELS = [
    "orange pen",
    "red handled screwdriver",
    "orange handled pliers",
    "blue pen",
    "hammer with wooden handle",
]

DETECTION_THRESHOLD = 0.35
MAX_BOXES = 1

ALPHA = 0.6
DISPLAY_SCALE = 1.0

USE_FP16 = True
CLEAN_CUDA_EACH_FRAME = True

SAVE_DIR = "_main_/outputs/camera_segmentation"


# ============================================================
# Utility
# ============================================================

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast_context(device, use_fp16=True):
    if device == "cuda" and use_fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def bgr_to_pil(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Model loading
# ============================================================

def load_models(device):
    print(f"Loading Grounding DINO: {DETECTION_MODEL_NAME}")
    bounding_box_processor = AutoProcessor.from_pretrained(DETECTION_MODEL_NAME)
    bounding_box_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTION_MODEL_NAME
    ).to(device)

    print(f"Loading SAM2: {SEGMENTATION_MODEL_NAME}")
    sam_processor = Sam2Processor.from_pretrained(SEGMENTATION_MODEL_NAME)
    sam_model = Sam2Model.from_pretrained(SEGMENTATION_MODEL_NAME).to(device)

    bounding_box_model.eval()
    sam_model.eval()

    return bounding_box_processor, bounding_box_model, sam_processor, sam_model


# ============================================================
# Detection
# ============================================================

def keep_top_k_detections(result, max_boxes):
    scores = result["scores"]

    if len(scores) <= max_boxes:
        return result

    top_indices = torch.topk(scores, k=max_boxes).indices

    filtered = {
        "boxes": result["boxes"][top_indices],
        "scores": result["scores"][top_indices],
        "labels": [result["labels"][int(i)] for i in top_indices],
    }

    return filtered


def detect_boxes(
    image_pil,
    text_labels,
    bounding_box_processor,
    bounding_box_model,
    device,
    threshold=0.35,
    max_boxes=6,
):
    inputs = bounding_box_processor(
        images=image_pil,
        text=[text_labels],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode(), autocast_context(device, USE_FP16):
        outputs_bb = bounding_box_model(**inputs)

    results_bb = bounding_box_processor.post_process_grounded_object_detection(
        outputs_bb,
        threshold=threshold,
        target_sizes=[(image_pil.height, image_pil.width)],
    )

    result = results_bb[0]

    result_cpu = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            result_cpu[key] = value.detach().cpu()
        else:
            result_cpu[key] = value

    result_cpu = keep_top_k_detections(result_cpu, max_boxes=max_boxes)

    del inputs
    del outputs_bb
    del results_bb

    if CLEAN_CUDA_EACH_FRAME:
        cleanup_cuda()

    return result_cpu


# ============================================================
# Segmentation
# ============================================================

def segment_boxes(
    image_pil,
    boxes,
    sam_processor,
    sam_model,
    device,
):
    if len(boxes) == 0:
        return None, None, None

    input_boxes = boxes.detach().cpu().numpy().astype(int)

    inputs_sam = sam_processor(
        images=image_pil,
        input_boxes=[input_boxes],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode(), autocast_context(device, USE_FP16):
        outputs_sam = sam_model(
            **inputs_sam,
            multimask_output=False,
        )

    masks = sam_processor.post_process_masks(
        outputs_sam.pred_masks,
        original_sizes=inputs_sam["original_sizes"],
    )

    masks_cpu = []
    for m in masks:
        if isinstance(m, torch.Tensor):
            masks_cpu.append(m.detach().cpu())
        else:
            masks_cpu.append(m)

    iou_scores = outputs_sam.iou_scores.detach().cpu()

    del inputs_sam
    del outputs_sam
    del masks

    if CLEAN_CUDA_EACH_FRAME:
        cleanup_cuda()

    return masks_cpu, iou_scores, input_boxes


# ============================================================
# Visualization
# ============================================================

def draw_segmentation_overlay(
    frame_bgr,
    masks,
    labels=None,
    boxes=None,
    scores=None,
    alpha=0.6,
):
    output = frame_bgr.copy()

    if masks is None:
        return output

    mask_data = masks[0]

    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.cpu().numpy()

    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    if boxes is not None:
        boxes = np.array(boxes).astype(int)

    if mask_data.ndim == 2:
        mask_data = mask_data[None, None, :, :]
    elif mask_data.ndim == 3:
        mask_data = mask_data[:, None, :, :]
    elif mask_data.ndim != 4:
        raise ValueError(f"Unexpected mask shape: {mask_data.shape}")

    num_objects, num_candidates, mask_h, mask_w = mask_data.shape

    frame_h, frame_w = output.shape[:2]

    overlay = output.copy()

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

        if obj_mask.shape[:2] != (frame_h, frame_w):
            obj_mask = cv2.resize(
                obj_mask.astype(np.uint8),
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        selected_masks.append(obj_mask)

        color = colors[i % len(colors)]
        overlay[obj_mask] = color

    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    for i, obj_mask in enumerate(selected_masks):
        mask_uint8 = obj_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(output, contours, -1, (255, 255, 255), 2)

        if labels is not None and i < len(labels):
            label_text = str(labels[i])
        else:
            label_text = f"object_{i + 1}"

        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i]
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
        font_scale = 0.55
        thickness = 2

        text_size, baseline = cv2.getTextSize(
            label_text,
            font,
            font_scale,
            thickness,
        )

        text_w, text_h = text_size

        rect_x1 = max(x_text, 0)
        rect_y1 = max(y_text - text_h - 6, 0)
        rect_x2 = min(x_text + text_w + 6, output.shape[1] - 1)
        rect_y2 = min(y_text + baseline + 2, output.shape[0] - 1)

        cv2.rectangle(
            output,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            label_text,
            (rect_x1 + 3, rect_y2 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return output


def draw_debug_info(frame, inference_time, num_detections, actual_fps):
    text_1 = f"Inference: {inference_time * 1000:.1f} ms"
    text_2 = f"Detections: {num_detections}"
    text_3 = f"Loop FPS: {actual_fps:.2f}"

    cv2.putText(frame, text_1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, text_2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, text_3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


# ============================================================
# Full frame pipeline
# ============================================================

def process_frame(
    frame_bgr,
    text_labels,
    bounding_box_processor,
    bounding_box_model,
    sam_processor,
    sam_model,
    device,
):
    image_pil = bgr_to_pil(frame_bgr)

    result_bb = detect_boxes(
        image_pil=image_pil,
        text_labels=text_labels,
        bounding_box_processor=bounding_box_processor,
        bounding_box_model=bounding_box_model,
        device=device,
        threshold=DETECTION_THRESHOLD,
        max_boxes=MAX_BOXES,
    )

    boxes = result_bb["boxes"]
    labels = result_bb["labels"]

    if len(boxes) == 0:
        return frame_bgr.copy(), result_bb

    masks, iou_scores, input_boxes = segment_boxes(
        image_pil=image_pil,
        boxes=boxes,
        sam_processor=sam_processor,
        sam_model=sam_model,
        device=device,
    )

    output_frame = draw_segmentation_overlay(
        frame_bgr=frame_bgr,
        masks=masks,
        labels=labels,
        boxes=input_boxes,
        scores=iou_scores,
        alpha=ALPHA,
    )

    del masks
    del iou_scores

    if CLEAN_CUDA_EACH_FRAME:
        cleanup_cuda()

    return output_frame, result_bb


# ============================================================
# Camera loop
# ============================================================

def run_camera():
    device = get_device()
    print(f"Using device: {device}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    (
        bounding_box_processor,
        bounding_box_model,
        sam_processor,
        sam_model,
    ) = load_models(device)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    target_dt = 1.0 / TARGET_FPS
    frame_id = 0

    print("Press 'q' to quit.")
    print("Press 's' to save current output frame.")

    try:
        while True:
            loop_start = time.perf_counter()

            ret, frame_bgr = cap.read()

            if not ret:
                print("Failed to read frame from camera.")
                break

            inference_start = time.perf_counter()

            output_frame, result_bb = process_frame(
                frame_bgr=frame_bgr,
                text_labels=TEXT_LABELS,
                bounding_box_processor=bounding_box_processor,
                bounding_box_model=bounding_box_model,
                sam_processor=sam_processor,
                sam_model=sam_model,
                device=device,
            )

            inference_time = time.perf_counter() - inference_start

            elapsed = time.perf_counter() - loop_start
            actual_fps = 1.0 / max(elapsed, 1e-6)

            output_frame = draw_debug_info(
                output_frame,
                inference_time=inference_time,
                num_detections=len(result_bb["boxes"]),
                actual_fps=actual_fps,
            )

            if DISPLAY_SCALE != 1.0:
                display_w = int(output_frame.shape[1] * DISPLAY_SCALE)
                display_h = int(output_frame.shape[0] * DISPLAY_SCALE)
                display_frame = cv2.resize(
                    output_frame,
                    (display_w, display_h),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                display_frame = output_frame

            cv2.imshow("Live Grounding DINO + SAM2", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                save_path = Path(SAVE_DIR) / f"camera_frame_{frame_id:06d}.png"
                cv2.imwrite(str(save_path), output_frame)
                print(f"Saved: {save_path}")

            frame_id += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, target_dt - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_camera()