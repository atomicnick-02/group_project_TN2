import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import Sam2Processor, Sam2Model
from gripper.src.visualizations import vis_bb, vis_seg
from torchvision.ops import nms

# -----------------------------
# Model loading
# -----------------------------
def load_models(device):
    print("Loading Grounding DINO...")
    gd_model_id = "IDEA-Research/grounding-dino-tiny"
    gd_processor = AutoProcessor.from_pretrained(gd_model_id)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(device)
    gd_model.eval()

    print("Loading SAM 2...")
    sam_model_id = "facebook/sam2.1-hiera-large"
    sam_processor = Sam2Processor.from_pretrained(sam_model_id)
    sam_model = Sam2Model.from_pretrained(sam_model_id).to(device)
    sam_model.eval()

    return gd_processor, gd_model, sam_processor, sam_model


# -----------------------------
# Detection (Grounding DINO)
# -----------------------------
def detect_boxes(
    image,
    text_labels,
    gd_processor,
    gd_model,
    device,
    threshold=0.30
):
    # Grounding DINO requires a single string prompt with periods
    text_prompt = " . ".join(text_labels) + " ."

    inputs = gd_processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gd_model(**inputs)

    # Post-process outputs to bounding boxes
    target_sizes = torch.tensor([image.size[::-1]]) # (height, width)
    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
		threshold=threshold,      # Changed from box_threshold        
		text_threshold=threshold,
        target_sizes=target_sizes
    )[0]

    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"]  # Returns list of strings matching the prompt

    # Apply Non-Maximum Suppression to remove overlapping boxes of the same object
    keep = nms(boxes.float(), scores.float(), 0.5)

    result_cpu = {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": [labels[i] for i in keep],
    }

    return result_cpu


# -----------------------------
# Segmentation (SAM 2)
# -----------------------------
def segment_boxes(
    image,
    boxes,
    sam_processor,
    sam_model,
    device
):
    input_boxes = boxes.detach().cpu().numpy().astype(int)

    inputs_sam = sam_processor(
        images=image,
        input_boxes=[input_boxes],
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs_sam = sam_model(
            **inputs_sam,
            multimask_output=False
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return masks_cpu, iou_scores, input_boxes


# -----------------------------
# Full pipeline for one image
# -----------------------------
def process_single_image(
    image_path,
    text_labels,
    gd_processor,
    gd_model,
    sam_processor,
    sam_model,
    device,
    threshold=0.30,
    alpha=0.6,
    scale=0.75,
    show_bb=False,
    show_seg=True,
    output_dir=None
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_path = Path(image_path)
    image = Image.open(str(image_path)).convert("RGB")

    print(f"\nProcessing image: {image_path}")

    result_bb = detect_boxes(
        image=image,
        text_labels=text_labels,
        gd_processor=gd_processor,
        gd_model=gd_model,
        device=device,
        threshold=threshold
    )

    if len(result_bb["boxes"]) == 0:
        print(f"No detections found for {image_path}")
        return None

    for box, score, label in zip(result_bb["boxes"], result_bb["scores"], result_bb["labels"]):
        box_list = [round(x, 2) for x in box.tolist()]
        print(f"Detected '{label}' with confidence {score.item():.3f} at {box_list}")

    if show_bb:
        vis_bb(
            image=image,
            result=result_bb,
            window_name=f"Bounding Boxes: {image_path.name}",
            scale=scale
        )

    masks, iou_scores, input_boxes = segment_boxes(
        image=image,
        boxes=result_bb["boxes"],
        sam_processor=sam_processor,
        sam_model=sam_model,
        device=device
    )

    save_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{image_path.stem}_segmented.png"

    segmented_img = vis_seg(
        masks=masks,
        image=image,
        labels=result_bb["labels"],
        boxes=input_boxes,
        scores=iou_scores,
        alpha=alpha,
        scale=scale,
        window_name=f"Segmented: {image_path.name}",
        save_path=save_path,
        show=show_seg
    )
    del masks
    del segmented_img

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "image_path": str(image_path),
        "save_path": str(save_path) if save_path is not None else None,
        "labels": list(result_bb["labels"]),
        "boxes": result_bb["boxes"].detach().cpu().numpy(),
        "scores": result_bb["scores"].detach().cpu().numpy(),
    }


# -----------------------------
# Full pipeline for many images
# -----------------------------
def process_multiple_images(
    image_paths,
    text_labels,
    threshold=0.30,
    alpha=0.6,
    scale=0.75,
    show_bb=False,
    show_seg=True,
    output_dir=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gd_processor, gd_model, sam_processor, sam_model = load_models(device)

    all_results = []

    for image_path in image_paths:
        result = process_single_image(
            image_path=image_path,
            text_labels=text_labels,
            gd_processor=gd_processor,
            gd_model=gd_model,
            sam_processor=sam_processor,
            sam_model=sam_model,
            device=device,
            threshold=threshold,
            alpha=alpha,
            scale=scale,
            show_bb=show_bb,
            show_seg=show_seg,
            output_dir=output_dir
        )
        all_results.append(result)

    return all_results


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    image_paths = [
        "gripper/data/agentview1.png",
    ]

    # Try simple labels first. DINO is much better at understanding basic shapes.
    text_labels = [
        "blue cube",
        "purple cube",
        "orange cube",
        "green cube",
    ]

    results = process_multiple_images(
        image_paths=image_paths,
        text_labels=text_labels,
        threshold=0.30,  # DINO's confidence scales differently than YOLO's. Start here.
        alpha=0.6,
        scale=1.5,
        show_bb=True,
        show_seg=True,
        output_dir="gripper/outputs/segmentation_results"
    )