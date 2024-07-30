# Emotion-Recognition

This project implements emotion recognition using a Vision Transformer (ViT) model. The implementation involves processing images from the RAF-DB dataset, training a ViT model, and evaluating its performance in classifying different emotions.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Data Augmentations](#data-augmentations)
- [Model Training](#model-training)
- [Evaluation](#evaluation)


## Introduction

Emotion recognition is a crucial task in computer vision with applications in various fields such as human-computer interaction, surveillance, and psychological analysis. This project leverages the power of Vision Transformer (ViT) models to classify emotions from images.

## Installation

To run this project, you'll need to have Python installed along with several libraries. The required libraries include `numpy`, `scikit-image`, `pillow`, `torch`, `torchvision`, `transformers`, `datasets`, and `matplotlib`.

## Dataset

The project uses the [RAF-DB](http://www.whdeng.cn/raf/model1.html) dataset, which contains images labeled with different emotions. You can download the dataset from the official RAF-DB website. Ensure that the dataset is placed in the appropriate directory for the project to access.

## Usage

1. **Load the dataset**: Load the RAF-DB dataset using the `datasets` library.
2. **Visualize Images**: Visualize random images from the dataset along with their labels to understand the data better.

## Data Augmentations

Data augmentation is a technique used to increase the diversity of the training data without actually collecting new data. This is achieved by applying various transformations to the existing data. In this project, several data augmentation techniques are applied to the images to improve the robustness and generalization ability of the ViT model:

1. **Rescaling**: Adjusting the pixel values to a common scale.
2. **Random Cropping**: Randomly selecting a portion of the image.
3. **Horizontal Flipping**: Flipping the image horizontally.
4. **Rotation**: Rotating the image by a random angle.
5. **Color Jitter**: Randomly changing the brightness, contrast, saturation, and hue of the image.
6. **Normalization**: Normalizing the image with the mean and standard deviation of the dataset.

These augmentations help in making the model invariant to various transformations and improve its performance on unseen data.

## Model Training

The training process involves the following steps:

1. **Initialize the Processor and Model**: Use the `transformers` library to initialize a Vision Transformer (ViT) model and processor.
2. **Set Training Arguments**: Define the training parameters such as learning rate, batch size, number of epochs, and evaluation strategy.
3. **Train the Model**: Use the `Trainer` class from the `transformers` library to train the ViT model on the RAF-DB dataset.

## Evaluation

After training the model, evaluate its performance on the test set. The model achieved an accuracy of 85%, demonstrating its effectiveness in classifying emotions from images.

