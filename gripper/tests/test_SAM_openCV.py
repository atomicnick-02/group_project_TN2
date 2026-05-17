import cv2
import torch
import numpy as np
import time
from PIL import Image
from transformers import Sam2Model, Sam2Processor

# --- CONFIGURATION SECTION ---
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
TARGET_FPS = 10         # SAM 2.1 Tiny can handle higher FPS
NUM_MASKS = 3           # SAM 2 outputs 3 masks by default
# ----------------------------

# 1. Initialize Model and Processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_list = ["facebook/sam2.1-hiera-tiny", "facebook/sam-vit-base","facebook/sam2.1-hiera-large"]
model_id = model_list[2]
# SAM 2 uses specific SAM 2 classes
model = Sam2Model.from_pretrained(model_id).to(device)
processor = Sam2Processor.from_pretrained(model_id)

# Optional: Performance boost for GPU
if device == "cuda":
    model = model.to(torch.float16)
    print("Model optimized for FP16.")

# 2. Setup Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

print(f"Running {model_id} on {device.upper()}")
prev_time = 0

while True:
    time_elapsed = time.time() - prev_time
    if time_elapsed < 1.0 / TARGET_FPS:
        continue
    prev_time = time.time()

    ret, frame = cap.read()
    if not ret: break

    # Prepare Image
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_frame)
    
    # SAM 2 Input Structure: [batch, objects, points, coordinates]
    h, w = frame.shape[:2]
    input_points = [[[[w // 2, h // 2]]]] 
    input_labels = [[[1]]] # 1 = positive click

    # 3. Inference
    inputs = processor(images=pil_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)
    
    # Ensure precision matches if using GPU
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Post-processing
    # SAM 2 post-processing is slightly different
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu()
    )[0] # Returns a list, take the first image
    
    output_frame = frame.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

    # 5. Mask Overlay (User defined count)
    # The mask tensor has shape [objects, mask_variations, height, width]
    for i in range(min(NUM_MASKS, masks.shape[1])):
        mask = masks[0, i].numpy().astype(bool)
        color = colors[i % len(colors)]
        
        overlay = output_frame.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)

    # UI and Interaction
    cv2.circle(output_frame, (w // 2, h // 2), 5, (255, 255, 255), -1)
    cv2.putText(output_frame, f"FPS: {1.0/time_elapsed:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("SAM 2.1 Tiny Live", output_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"sam2_capture_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        cv2.imwrite(f"analysis_{filename}", output_frame)
        print(f"Files saved: {filename}")

cap.release()
cv2.destroyAllWindows()