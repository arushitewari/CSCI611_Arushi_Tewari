from ultralytics import YOLO
import os

model = YOLO("/Users/arushi_tewari/runs/detect/train-2/weights/best.pt")

test_folder = os.path.expanduser("~/Traffic-and-Road-Signs-1/test/images")

results = model(test_folder, conf=0.5, save=True)

print("Done! Results saved.")