# slash
Slash's AI internship task notebook

# Deep Learning Image Classification Notebook

## Overview

This notebook contains code for training a deep learning model for image classification task for Slash's AI internship using Convolutional Neural Networks (CNNs). The model is trained on a dataset containing images of 3 classes, Fashion, Home and Nutrition, and it is capable of predicting the class of an input image with a certain level of accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)


## Introduction

In this notebook, we use deep learning techniques, specifically CNNs, to perform image classification.

## Dataset

The dataset used for training and evaluation consists of images from Slash's mobile application and some web-scraped images belonging to each class. Each image is labeled with its corresponding class, allowing the model to learn the relationships between images and their associated labels.

## Data Augmentation

To improve the robustness and generalization of the model, data augmentation techniques are applied to the training dataset. These techniques involve applying random transformations such as rotation, flipping, brightness, zoom and scaling to the input images. This helps the model learn to recognize objects from various perspectives and orientations and helps mitigate the small size of the data, making it more resilient to variations in the input data.

## Model Architecture

The model architecture includes convolutional layers followed by max-pooling layers and batch normalization for feature extraction. The final layers consist of fully connected layers with softmax activation for classification. This architecture ensures effective feature learning and robust classification performance.

## Training

The model is trained using the augmented training dataset with the RMSprop optimizer and categorical cross-entropy loss function. Early stopping and learning rate reduction techniques are employed to prevent overfitting and improve generalization.

## Evaluation

The trained model is evaluated on a separate validation dataset to assess its performance and generalization ability. Metrics such as accuracy are used to measure the model's effectiveness in classifying unseen images.
