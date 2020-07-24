# Speech Emotion Recognition (SER)

Recognizing emotion from speech with artificial neural network. 

## Overview

This repository contains my implementation of a complete SER model training and evaluation in pyTorch, from features extraction to model cross-validation. It may hopefully serve as a framework for SER model evaluation, where new models can be plugged in easily.

I also included resources for speech emotion recognition, including publications, datasets, and useful python packages. These resources are collected while working on SER project as part of my MSc. study. 

## This Repository (pyTorch SER Implementation)

### Features extraction

### Models

### Training and Evaluation


## SER Publications and Results

### "Leaderboard"

### Papers


## SER Datasets

### IEMOCAP

### emoDB


## Useful Packages


## Neural Network Model Training




This package contains pyTorch implementation of speech emotion recognition models. The models are based on fully-convolutional network and convolutional-recurrent network. Available models are:

1. __AlexNet__ 
- Convolutional neural network architecture for [image classification](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- Finetuned to classify emotion from speech spectrograms. 
- The pre-trained model is taken from *torchvision* models in pyTorch.
2. __Fully-Convolutional Network with Attention (FCN+Attention)__
- Inspired by the work of Zhang et. al. (2019) presented in [*Attention Based Fully Convolutional Network for Speech Emotion Recognition*](https://arxiv.org/abs/1806.01506). 
- pyTorch implementation by Samuel Samsudin Ng.
3. __3D Convolutional Recurrent Network with Attention (3D ACRNN)__
- Based on the work of Chen et. al. (2018) presented in [*3-D Convolutional Recurrent Neural Networks With Attention Model for Speech Emotion Recognition*](https://www.researchgate.net/publication/326638635_3-D_Convolutional_Recurrent_Neural_Networks_With_Attention_Model_for_Speech_Emotion_Recognition) . 
- TensorFlow implementation by the authors can be found [here](https://github.com/xuanjihe/speech-emotion-recognition).
- pyTorch implementation by Hoang Nghia Tuyen can be found [here](https://github.com/NTU-SER/speech_utils). 
- The model is then integrated into this package.
