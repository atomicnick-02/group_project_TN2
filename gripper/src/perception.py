from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers import Sam2Processor, Sam2Model


DETECTION_MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
SEGMENTATION_MODEL_NAME = "facebook/sam2.1-hiera-tiny"
DETECTION_THRESHOLD = 0.35
MAX_BOXES_PER_LABEL = 1
USE_FP16 = True


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast_context(device):
    if device == "cuda" and USE_FP16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class VLMPerception:
    """
    Grounding DINO + SAM2 perception module.
    Accepts a numpy (H, W, 3) RGB image and a list of text labels,
    returns bounding boxes, masks, scores, and labels.
    """

    def __init__(self, device=None):
        self.device = device or get_device()
        print(f"[VLMPerception] Loading models on {self.device} ...")

        self.bb_processor = AutoProcessor.from_pretrained(DETECTION_MODEL_NAME)
        self.bb_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            DETECTION_MODEL_NAME
        ).to(self.device).eval()

        self.sam_processor = Sam2Processor.from_pretrained(SEGMENTATION_MODEL_NAME)
        self.sam_model = Sam2Model.from_pretrained(SEGMENTATION_MODEL_NAME).to(
            self.device
        ).eval()

        print("[VLMPerception] Models loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_rgb: np.ndarray, text_labels: list[str]) -> dict:
        """
        Args:
            image_rgb: (H, W, 3) uint8 numpy array in RGB
            text_labels: list of strings, e.g. ["red cube", "blue cube"]

        Returns dict with keys:
            boxes       : (N, 4) int array  [x1, y1, x2, y2]
            labels      : list[str] of length N
            scores      : (N,) float array
            masks       : (N, H, W) bool array  (None if no detections)
        """
        image_pil = Image.fromarray(image_rgb)

        detection = self._detect(image_pil, text_labels)

        boxes = detection["boxes"]   # cpu tensor
        labels = detection["labels"]
        scores = detection["scores"]

        if len(boxes) == 0:
            return {
                "boxes": np.empty((0, 4), dtype=int),
                "labels": [],
                "scores": np.empty((0,)),
                "masks": None,
            }

        masks = self._segment(image_pil, boxes)  # (N, H, W) bool or None

        return {
            "boxes": boxes.numpy().astype(int),
            "labels": labels,
            "scores": scores.numpy(),
            "masks": masks,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect(self, image_pil: Image.Image, text_labels: list[str]) -> dict:
        inputs = self.bb_processor(
            images=image_pil,
            text=[text_labels],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode(), autocast_context(self.device):
            outputs = self.bb_model(**inputs)

        results = self.bb_processor.post_process_grounded_object_detection(
            outputs,
            threshold=DETECTION_THRESHOLD,
            target_sizes=[(image_pil.height, image_pil.width)],
        )[0]

        # Move to CPU
        result_cpu = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in results.items()
        }

        # Keep only top-k per label
        result_cpu = self._top_k_per_label(result_cpu, k=MAX_BOXES_PER_LABEL)

        del inputs, outputs, results
        self._cleanup()
        return result_cpu

    def _segment(self, image_pil: Image.Image, boxes: torch.Tensor) -> np.ndarray | None:
        input_boxes = boxes.cpu().numpy().astype(int)

        inputs = self.sam_processor(
            images=image_pil,
            input_boxes=[input_boxes],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode(), autocast_context(self.device):
            outputs = self.sam_model(**inputs, multimask_output=False)

        masks_list = self.sam_processor.post_process_masks(
            outputs.pred_masks,
            original_sizes=inputs["original_sizes"],
        )

        # masks_list[0]: tensor (N, 1, H, W)
        mask_tensor = masks_list[0].detach().cpu()  # (N, 1, H, W)
        masks = mask_tensor[:, 0, :, :].numpy().astype(bool)  # (N, H, W)

        del inputs, outputs, masks_list
        self._cleanup()
        return masks

    @staticmethod
    def _top_k_per_label(result: dict, k: int) -> dict:
        """Keep at most k detections per unique label."""
        labels = result["labels"]
        scores = result["scores"]
        boxes = result["boxes"]

        from collections import defaultdict
        label_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            label_indices[lbl].append(i)

        keep = []
        for lbl, indices in label_indices.items():
            top = sorted(indices, key=lambda i: scores[i], reverse=True)[:k]
            keep.extend(top)

        keep = sorted(keep)
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": [labels[i] for i in keep],
        }

    @staticmethod
    def _cleanup():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()