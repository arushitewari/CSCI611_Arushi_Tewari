from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

data_yaml = os.path.expanduser("~/Traffic-and-Road-Signs-1/data.yaml")

model.train(
    data=data_yaml,
    epochs=10,
    imgsz=640,
    batch=8,
    augment=True
)

print("Training done!")
