# CSCI611 - Assignment 4: Style Transfer with Deep Neural Networks

**Name:** Arushi Tewari  
**Course:** CSCI 611 - Spring 2026  
**Assignment:** A4 - Neural Style Transfer

---

## Overview

This project implements neural style transfer based on the paper [Image Style Transfer Using Convolutional Neural Networks by Gatys et al. (2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

A pre-trained VGG19 network extracts content and style features from two input images, then iteratively updates a target image to minimize both content and style losses — producing an image that looks like the content photo painted in the style of the style image.

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Upload `Style_Transfer_Exercise.ipynb`
3. Enable GPU: **Runtime → Change runtime type → T4 GPU**
4. Click **Runtime → Run all**

### Option 2: Local Machine
Install dependencies first:
```bash
pip3 install torch torchvision matplotlib numpy pillow requests
```
Then open the notebook:
```bash
jupyter notebook Style_Transfer_Exercise.ipynb
```

---

## Images Used

| Image | Description |
|-------|-------------|
| Content | NYC skyline photograph (loaded from URL) |
| Style | Floral still-life painting (loaded from URL) |

Images are loaded automatically from URLs inside the notebook — no manual download needed.

---

## Hyperparameter Experiments

Four experiments were conducted by changing the weights cell in the notebook:

| Experiment | style_weight | Layer Weights | Effect |
|------------|-------------|---------------|--------|
| 1 (Default) | 1e6 | Early layers emphasized | Balanced style and content |
| 2 | 1e8 | Early layers emphasized | Heavy stylization |
| 3 | 1e4 | Early layers emphasized | Subtle stylization |
| 4 | 1e6 | Later layers emphasized | Fine-grained style details |

To reproduce each experiment, change the following cell in the notebook:
```python
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}
content_weight = 1
style_weight = 1e6
steps = 2000
```

---

## File Structure

```
Assignment_4/
├── Style_Transfer_Exercise.ipynb   # Main notebook with all TODOs completed
├── README.md                       # This file
└── CSCI611_A4_Report.pdf          # Assignment report
```
