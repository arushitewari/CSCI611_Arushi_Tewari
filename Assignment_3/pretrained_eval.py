from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

data_yaml = os.path.expanduser("~/Traffic-and-Road-Signs-1/data.yaml")

metrics = model.val(data=data_yaml)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)