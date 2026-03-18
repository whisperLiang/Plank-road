import cv2
import torch
from model_management.model_zoo import build_detection_model
from model_management.model_split import extract_backbone_features

model = build_detection_model("fasterrcnn_mobilenet_v3_large_fpn", False)
model.eval()
cap = cv2.VideoCapture("./video_data/road.mp4")
ret, frame = cap.read()
if not ret: print("No frame")
else:
    try:
        res = extract_backbone_features(model, frame)
        print("Success")
    except Exception as e:
        print(f"Error: {e}")
