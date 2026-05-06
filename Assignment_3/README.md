# CSCI611 - Assignment 3: Small Object Detection Using YOLO

**Name:** Arushi Tewari  
**Course:** CSCI 611 - Spring 2026  
**Assignment:** A3 - Small Object Detection Using YOLO

---

## Overview

This project implements small object detection using YOLOv8 to detect traffic signs from images. A pre-trained YOLOv8n model is first evaluated as a baseline, then fine-tuned on a traffic sign dataset to significantly improve detection performance.

---

## Dataset

**Traffic and Road Signs Dataset**  
Source: https://universe.roboflow.com/usmanchaudhry622-gmail-com/traffic-and-road-signs  
- 10,000 images, 29 traffic sign classes, pre-annotated in YOLO format

---

## Requirements

Install dependencies:

```bash
pip3 install ultralytics opencv-python numpy torch torchvision torchaudio roboflow
```

---

## How to Run

### Step 1: Download the Dataset
```bash
python3 download_dataset.py
```
This will download the dataset into a folder called `Traffic-and-Road-Signs-1`.

### Step 2: Test Pre-trained YOLO (Baseline)
```bash
python3 pretrained_test.py
```
Runs the pre-trained YOLOv8n model on the test images. Results saved to `runs/detect/predict`.

### Step 3: Evaluate Pre-trained Model Metrics
```bash
python3 pretrained_eval.py
```
Prints mAP, Precision, and Recall for the pre-trained model.

### Step 4: Fine-tune YOLO on Traffic Signs
```bash
python3 train_model.py
```
Fine-tunes YOLOv8n for 10 epochs on the traffic sign dataset. Trained weights saved to `runs/detect/train-2/weights/best.pt`.

> Note: Training takes approximately 2-3 hours on CPU.

### Step 5: Test Fine-tuned Model
```bash
python3 final_test.py
```
Runs the fine-tuned model on test images. Results saved to `runs/detect/predict-2`.

---

## Results Summary

| Metric | Pre-trained | Fine-tuned |
|--------|-------------|------------|
| mAP@50 | 0.56% | 93.3% |
| mAP@50-95 | 0.25% | 77.9% |
| Precision | 1.14% | 91.0% |
| Recall | 15.17% | 93.4% |

---

## File Structure

```
Assignment_3/
├── download_dataset.py      # Downloads dataset from Roboflow
├── pretrained_test.py       # Runs pre-trained YOLO on test images
├── pretrained_eval.py       # Evaluates pre-trained model metrics
├── train_model.py           # Fine-tunes YOLO on traffic sign dataset
├── final_test.py            # Runs fine-tuned model on test images
├── best.pt                  # Trained model weights
├── README.md                # This file
└── CSCI611_A3_Report.pdf    # Assignment report
```
