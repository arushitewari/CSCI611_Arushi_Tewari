from ultralytics import YOLO
import os

model = YOLO("/Users/arushi_tewari/runs/detect/train-2/weights/best.pt")
test_folder = os.path.expanduser("~/Traffic-and-Road-Signs-1/test/images")

# Count detections at conf=0.3
results_03 = model(test_folder, conf=0.3, verbose=False)
count_03 = sum(len(r.boxes) for r in results_03)

# Count detections at conf=0.7
results_07 = model(test_folder, conf=0.7, verbose=False)
count_07 = sum(len(r.boxes) for r in results_07)

print(f"Total detections at conf=0.3: {count_03}")
print(f"Total detections at conf=0.7: {count_07}")
