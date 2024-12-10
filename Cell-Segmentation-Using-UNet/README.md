# Cell Segmentation Using U-Net Architecture

## Overview

This project demonstrates the application of the **U-Net architecture** for segmenting cell nuclei from microscopy images. Nuclei segmentation is a vital task in biomedical image analysis, assisting in understanding cell morphology and disease pathology. The dataset used is from the 2018 Data Science Bowl, which contains annotated images and their corresponding masks. The project covers data preprocessing, model implementation, training, and evaluation to achieve accurate segmentation results.

## Table of Contents

- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [U-Net Architecture](#u-net-architecture)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run the Project](#how-to-run-the-project)
- [Future Work](#future-work)

---

## Dataset

The dataset is sourced from the [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018). It contains:
- Microscopy images in various dimensions and formats.
- Ground truth masks corresponding to cell nuclei in each image.

**Preprocessing Steps:**
- Images and masks were resized to 128x128 pixels.
- Combined multiple masks for a single image into one unified mask.

---

## Project Workflow

1. **Dataset Setup**:
   - Downloaded the dataset using Kaggle API.
   - Extracted the data for preprocessing.

2. **Preprocessing**:
   - Loaded images and masks.
   - Resized and normalized the data to ensure consistency.

3. **Model Implementation**:
   - Built the U-Net model with an encoder-decoder structure and skip connections.

4. **Training**:
   - Used a 90/10 training-validation split.
   - Trained the model using `Adam` optimizer and binary cross-entropy loss.

5. **Evaluation**:
   - Visualized predictions on training data.
   - Plotted training and validation accuracy/loss metrics.

---

## U-Net Architecture

The U-Net model is designed specifically for image segmentation. It has:
- **Encoder**: Extracts features at multiple levels using convolutional layers and max-pooling.
- **Decoder**: Reconstructs the segmentation map by upsampling features and using skip connections from the encoder.
- **Output**: A single-channel binary mask generated using a sigmoid activation function.

---

## Results

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~90%
- Visualizations of predicted masks demonstrate the model's capability to accurately delineate nuclei boundaries.
