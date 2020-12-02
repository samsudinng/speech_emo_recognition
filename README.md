# Speech Emotion Recognition (SER) with Pre-trained Image Classifier - PyTorch Implementation

This repository containts my works on speech emotion recognition using spectrogram of the utterances as input. Spectrogram contains time-frequency information which reflects the acoustic cues, including those from emotion. I formulated the task as image classification using pre-trained image classification network. I proposed two AlexNet-based models: __AlexNet Finetuning__ and __FCN+GAP__ (Fully-Convolutional Network with Global Average Pooling). These networks were finetuned for speech emotion recognition with __IEMOCAP__ emotion speech dataset. I also implemented two deep learning enhancement techniques based on __mixup augmentation__ and __stability training__. These enhancements aim to improve generalization and robustness of the models given unseen utterances and diverse acoustic environments. 

This is the topic of my project as part of my post-graduate study (MSc. in AI, Nanyang Technological University, Singapore). This repository also includes resources for SER, including publications, datasets, and useful python packages. These resources are collected during my research, implementation and optimization phase. 


## This Repository

My approach was inspired by the work of Zhang et. al. (2019) presented in [*Attention Based Fully Convolutional Network for Speech Emotion Recognition*](https://arxiv.org/abs/1806.01506). The pretrained image classifier model was based on [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), a CNN architecture designed for image classification. The following sections describes the structure and components of this framework.

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

Two models are available in the framework:  

1. __AlexNet Finetuning__ 
- Pre-trained AlexNet, finetuned to classify emotion from speech spectrogram images (IEMOCAP). 
- The model is based on *torchvision.models.alexnet* model in pyTorch.

2. __FCN+GAP__
- Pre-trained AlexNet with the fully connected layers replaced with global average pooling (GAP).
- Finetuned to classify emotion from speech spectrogram images (IEMOCAP)
- Fully-connected layers are prone to over-fitting and require large number of parameters. GAP was proposed by Lin et. al. (2014) in [*Network In Network*](https://arxiv.org/abs/1312.4400) to address these issues.
- This model perform as well as AlexNet Finetuning but requiring only 4.5% as many parameters.


The model to be trained can be selected via command line with the following labels. The summary of model parameters and accuracy (5-fold, speaker independent cross-validation) are also summarized below.

|Model label|Model Name|# of Params.|Weighted Accuracy|Unweighted Accuracy| Model Setting |
|-----------|----------|----------|----------|----------| ----------|
|*'alexnet'*|AlexNet Finetuning| ~57m | 74.0% | 64.4%| baseline + stability training|
|*'alexnet_gap'*|FCN+GAP| ~2.5m | 73.2% | 62.6% | baseline |


------------------------------------
## Usage

### Feature Extraction

### Training/Finetuning
The main training script is train_ser.py.


python train_ser.py <*features_file*> --ser_model <*model*> --val_id <*vid*>
 --test_id <*tid*> --num_epochs <*n*> --batch_size <*b*> --lr <*l*> --seed <*s*> --gpu <*g*> --save_label <*label*> *{additional flags}*
 
 Example:
 
 ```python
python train_ser.py IEMOCAP_logspec200.pkl --ser_model alexnet --val_id 1F --test_id 1M --num_epochs 100 --batch_size 64 --lr 1e-3 --seed 111 --gpu 1 --save_label alexnet_baseline --pretrained --mixup
 ```
 
|Script Parameter|Remarks|
|-----------|----------|
|features_file|File containing the features (spectrogram) extracted from the speech utterance|
|model|alexnet, alexnet_gap|
|vid| the ID of the speaker to be used as validation set (see note)|
|tid| the ID of the speaker to be used as test set (see note)|
|g|0 (cpu), 1 (gpu)|
|label|the best finetuned model will be save to label.pth|
|additional flags|--pretrained (to use pre-trained model)|
| | --mixup (to use mixup)|
| | --oversampling (to use random dataset oversampling)|

Note:
- IEMOCAP database consists of 5 sessions * 2 speakers per session. The speakers (5 males, 5 females) have been assigned ID based on the session and gender {1F, 1M, 2F, 2M, 3F, 3M, 4F, 4M, 5F, 5M}

## Requirements

------------------------------------
## SER Publications


------------------------------------
## SER Datasets

### IEMOCAP

### emoDB

------------------------------------
## Useful Packages

### Audio feature extraction

librosa, python_speech_features

### 

------------------------------------
## How to Improve Model Performance?

### Dataset Oversampling

### Mixup

### Data Augmentation






