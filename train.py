import torch
from ultralytics import YOLO

model = YOLO("yolov5s.pt")
model.train(
    data="weapon.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    name="weapon-detector"
)