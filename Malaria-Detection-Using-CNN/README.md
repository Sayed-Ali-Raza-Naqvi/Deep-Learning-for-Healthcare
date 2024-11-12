# Malaria Detection Using Convolutional Neural Networks (CNN)

This project demonstrates the use of Convolutional Neural Networks (CNN) to classify malaria-infected cells in microscopy images. The model is trained on a dataset of blood cell images, with two categories: parasitized (malaria-infected) and uninfected. The goal is to create a model that can accurately identify malaria-infected cells from images.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Training the Model](#training-the-model)
7. [Evaluation and Metrics](#evaluation-and-metrics)
8. [Results](#results)
9. [License](#license)

## Overview

This repository provides the necessary code to train a CNN model for malaria detection. It covers the following steps:

- **Data Preprocessing**: Includes dataset extraction, image resizing, and normalization.
- **Model Architecture**: A CNN model designed to classify images into two categories (parasitized or uninfected).
- **Training**: The model is trained with real-time data augmentation and early stopping.
- **Evaluation**: The model is evaluated using common classification metrics like accuracy, precision, recall, AUC, ROC curve, and confusion matrix.
- **Model Saving**: The trained model is saved for later use.

## Dataset

The dataset used in this project is the **Cell Images for Detecting Malaria** dataset, available on Kaggle. The dataset contains images of blood cells classified into two categories:

- **Parasitized**: Malaria-infected cells
- **Uninfected**: Healthy cells

You can download the dataset from Kaggle using the following link:

[Kaggle Dataset - Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

After downloading, extract the dataset and place it in the working directory.

## Installation

Before running the project, make sure to install the necessary libraries. The required libraries are listed below:

- **TensorFlow/Keras**: For building and training the CNN model
- **OpenCV**: For image processing
- **NumPy**: For numerical operations
- **Matplotlib**: For visualization
- **scikit-learn**: For model evaluation metrics

Additionally, download the dataset from Kaggle and unzip it into the project directory.

## Usage

1. **Prepare the Dataset**: Download and extract the dataset to a folder in your project directory.
2. **Model Training**: Run the provided script to start training the CNN model. The script handles the entire pipeline: data loading, preprocessing, model building, training, and evaluation.
3. **Evaluate the Model**: The model will be evaluated on the test set, and metrics like accuracy, precision, recall, and AUC will be printed out.
4. **Visualize Results**: After training, the script will generate visualizations such as the accuracy/loss plot, ROC curve, precision-recall curve, and confusion matrix.
5. **Save the Model**: The trained model will be saved in the `.keras` format for future use.

## Model Architecture

The model used in this project is a Convolutional Neural Network (CNN) with the following architecture:

1. **Input Layer**: Accepts images of size 64x64x3 (RGB).
2. **Convolutional Layers**: Two convolutional layers with 32 filters, each followed by MaxPooling and BatchNormalization.
3. **Dropout Layers**: To reduce overfitting, dropout layers are added after convolutional layers and fully connected layers.
4. **Fully Connected Layers**: After flattening the output of the convolutional layers, two fully connected layers (with 512 and 256 neurons) are used.
5. **Output Layer**: A single neuron with a sigmoid activation function for binary classification (malaria-infected or not).

## Training the Model

The training process involves the following steps:

- **Data Augmentation**: To improve generalization, the training images undergo real-time augmentation using random rotations, shifts, shears, zooms, and flips.
- **Optimization**: The Adam optimizer is used with a learning rate of 0.0001.
- **Early Stopping**: The model stops training if the validation loss doesn't improve for 5 epochs.
- **Learning Rate Reduction**: If the validation loss plateaus, the learning rate is reduced by half.

The training runs for a maximum of 30 epochs, but early stopping ensures that training can halt earlier if the model reaches its best performance.

## Evaluation and Metrics

The model's performance is evaluated on the test set using the following metrics:

- **Accuracy**: Percentage of correct predictions.
- **Precision**: The ratio of true positive predictions to all positive predictions.
- **Recall**: The ratio of true positive predictions to all actual positive instances.
- **AUC (Area Under the ROC Curve)**: A measure of the model's ability to distinguish between the two classes.
- **Confusion Matrix**: A matrix that shows the count of true positive, false positive, true negative, and false negative predictions.
- **ROC Curve**: A plot showing the trade-off between true positive rate and false positive rate.
- **Precision-Recall Curve**: A plot showing the trade-off between precision and recall.

## Results

After training, the model’s performance can be assessed through various metrics and plots:

1. **Training and Validation Accuracy/Loss Plots**: Show how the model's accuracy and loss evolve during training.
2. **ROC Curve**: A plot of the true positive rate vs. the false positive rate, showing the model’s discriminative ability.
3. **Precision-Recall Curve**: Helps evaluate the model's performance in terms of precision and recall.
4. **Confusion Matrix**: Visualizes the number of correct and incorrect predictions across both classes.
