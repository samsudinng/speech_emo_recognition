# Speech Emotion Recognition (SER) Framework with PyTorch

PyTorch implementation of a __Fully-Convolutional Network with Attention (FCN+Attention)__ as a topic for my MSc. in AI (Nanyang Technological University, Singapore) Master Project. As I built the model and training framework, I added support for various dataset, features, SER models, training techniques and cross-validation. It may hopefully serves as a framework for other SER model implementation, where new models can be plugged in and evaluated quickly.   

I also included resources for SER, including publications, datasets, and useful python packages. These resources are collected during my research, implementation and optimization phase. 


## This Repository

I implemented a fully-convolution network as my main model. It was inspired by the work of Zhang et. al. (2019) presented in [*Attention Based Fully Convolutional Network for Speech Emotion Recognition*](https://arxiv.org/abs/1806.01506). The model was based on [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), a CNN architecture designed for image classification. The following sections describes the structure and components of this framework.

### Features extraction

The framework supports features extraction from the following database:

1. [__IEMOCAP__](https://sail.usc.edu/iemocap/), a database of approximately 12 hours of audiovisual data, including video, speech, motion capture of face, and text transcriptions. The database was organized into 5 sessions, where each session contains dialog between 2 unique actors (1 male, 1 female). Each session contains scripted and improvised dialog, manually segmented into utterances and annotated by at least 3 human annotators. The average duration of each utterance is 4-5 sec. The emotion classes are {*anger, happiness, excitement, sadness, frustration, fear, surprise, neutral, others*}.

2. [__emoDB__](http://www.emodb.bilderbar.info/start.html), a German database of emotional speech of approximately 500 utterances. The utterances were recorded with 10 actors (5 male, 5 female). The average duration of each utterance is 2-3 sec. The emotion classes are {*happy, angry, anxious, fearful, bored, disgust, neutral*}.

Spectral features are extracted from the dataset using [*librosa*](https://librosa.org) audio analysis package. The supported features are:

|Features Label|Features|# of channels|
|-----|--------|-------------|
|*'logspec'* |Log spectrogram |1 |
|*'logmelspec'* |Log Mel spectrogram |1 |
|*'logmeldeltaspec'* |Log Mel spectrogram, ∆ , ∆∆ |3 |
|*'lognrevspec'* |Log spectrogram of signal, signal+white noise and signal+reverb |3 |

The spectrogram of each utterance is splitted into segments with length *T*. If the length of the spectrogram or last block of the split is < *T*, the spectrogram is padded with 0s. Each segment has a shape of (*C, F, T*) where *C* is the number of feature channels, and *F* the number of frequency or mel frequency bins. The extracted features are then organized into a dictionary with speaker ID as the keys, and tuple of all spectrogram segments, utterance-level labels, segment-level labels, and number of segments per utterance corresponding to each speakers.

        {'speaker_id': (all_spectrograms_segments, all_utterance_labels, all_segments_labels, number_of_segments_per_utterance)}


### SER Models

Three models are available in the framework:  

1. __FCN+Attention__
- AlexNet with the fully connected layers replaced with Attention layer and output layer.
- Batchnorm layers are added at the input layer and at the output of each CNN layer.

2. __AlexNet__ 
- Finetuned to classify emotion from speech spectrograms. 
- The model is based on *torchvision.models.alexnet* model in pyTorch.

3. __3D Convolutional Recurrent Network with Attention (3D ACRNN)__
- Based on the work of Chen et. al. (2018) presented in [*3-D Convolutional Recurrent Neural Networks With Attention Model for Speech Emotion Recognition*](https://www.researchgate.net/publication/326638635_3-D_Convolutional_Recurrent_Neural_Networks_With_Attention_Model_for_Speech_Emotion_Recognition) . 
- TensorFlow implementation by the authors can be found [here](https://github.com/xuanjihe/speech-emotion-recognition).
- pyTorch implementation by Hoang Nghia Tuyen can be found [here](https://github.com/NTU-SER/speech_utils). 


The model to be trained can be selected via command line with the following labels.

|Model label|Model Name|
|-----------|----------|
|*'fcn_attention'*|FCN+Attention|
|*'alex_net'*|AlexNet|
|*'3d_acrnn'*|#3D ACRNN|

### Training and Evaluation



## Usage

## Requirements


## SER Publications and Results

### "Leaderboard"

### Papers


## SER Datasets

### IEMOCAP

### emoDB


## Useful Packages


## Neural Network Model Training: Tips and Tricks





