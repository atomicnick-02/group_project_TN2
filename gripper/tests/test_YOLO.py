import cv2
import torch
from ultralytics import YOLOE

# ---------------------------------------------------------
# Text labels / prompts
# ---------------------------------------------------------
TEXT_LABELS = [
    "orange pen",
    "red handled screwdriver",
    "orange handled pliers",
    "blue pen",
    "hammer with wooden handle",
]

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
MODEL_NAME = "yoloe-26s-seg.pt"   # good starting point
CAMERA_ID = 0                     # 0 = default webcam
CONF_THRES = 0.25
IOU_THRES = 0.50
IMG_SIZE = 640

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------
# Load YOLOE segmentation model
# ---------------------------------------------------------
model = YOLOE(MODEL_NAME)

# Set text prompts only once
model.set_classes(TEXT_LABELS)

# ---------------------------------------------------------
# Open webcam
# ---------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing CAMERA_ID to 1 or 2.")

# Optional camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from webcam.")
        break

    # -----------------------------------------------------
    # YOLOE prediction
    # -----------------------------------------------------
    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False,
    )

    result = results[0]

    # -----------------------------------------------------
    # Draw segmentation masks, boxes, labels using Ultralytics
    # -----------------------------------------------------
    annotated_frame = result.plot()

    # -----------------------------------------------------
    # Optional: print detections to terminal
    # -----------------------------------------------------
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = result.names[cls_id]
            print(f"Detected: {label} | confidence: {conf:.3f}")

    # -----------------------------------------------------
    # Show using OpenCV
    # -----------------------------------------------------
    cv2.imshow("YOLOE live segmentation", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()