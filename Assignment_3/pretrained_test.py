from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")  # downloads pretrained model automatically

test_folder = os.path.expanduser("~/Traffic-and-Road-Signs-1/test/images")

results = model(test_folder, conf=0.5, save=True)

print("Done! Check the runs/detect folder for results.")