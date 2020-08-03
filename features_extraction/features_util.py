import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import os
from collections import defaultdict
from tqdm import tqdm
from pysndfx import  AudioEffectsChain


def extract_features(speaker_files, features, params):
    
    speaker_features = defaultdict()
    for speaker_id in tqdm(speaker_files.keys()):
        
        data_tot, labels_tot, labels_segs_tot, segs = list(), list(), list(), list()

        for wav_path, emotion in speaker_files[speaker_id]:
            
            # Read wave data
            x, sr = librosa.load(wav_path, sr=None)

            # Apply pre-emphasis filter
            x = librosa.effects.preemphasis(x, zi = [0.0])

            # Extract required features into (C,F,T)
            features_data = GET_FEATURES[features](x, sr, params)

            # Segment features into (N,C,F,T)
            features_segmented = segment_nd_features(features_data, emotion, params['segment_size'])

            #Collect all the segments
            data_tot.append(features_segmented[1])
            labels_tot.append(features_segmented[3])
            labels_segs_tot.extend(features_segmented[2])
            segs.append(features_segmented[0])

        # Post process
        data_tot = np.vstack(data_tot).astype(np.float32)
        labels_tot = np.asarray(labels_tot, dtype=np.int8)
        labels_segs_tot = np.asarray(labels_segs_tot, dtype=np.int8)
        segs = np.asarray(segs, dtype=np.int8)

        # Make sure everything is extracted properly
        assert len(labels_tot) == len(segs)
        assert data_tot.shape[0] == labels_segs_tot.shape[0] == sum(segs)

        #Put into speaker features dictionary
        speaker_features[speaker_id] = (data_tot, labels_tot, labels_segs_tot, segs)
    
    assert len(speaker_features) == len (speaker_files)

    return speaker_features

def extract_logspec(x, sr, params):
    
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    nfreq         = params['nfreq']

    #calculate stft
    spec = np.abs(librosa.stft(x, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec =  librosa.amplitude_to_db(spec, ref=np.max)
    
    #extract the required frequency bins
    spec = spec[:nfreq]
    
    #Shape into (C, F, T), C = 1
    spec = np.expand_dims(spec,0)

    return spec


def extract_lognrevspec(x, sr, params):
    
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    nfreq         = params['nfreq']

    #Input + gaussian white noise
    noise=np.random.normal(0, 0.002, x.shape[0])
    x_noisy = x+noise
    
    #Input + reverb
    fx = (AudioEffectsChain().reverb())
    x_reverb = fx(x)
    
    #calculate stft
    spec = np.abs(librosa.stft(x, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec =  librosa.amplitude_to_db(spec, ref=np.max)

    spec_noise = np.abs(librosa.stft(x_noisy, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec_noise =  librosa.amplitude_to_db(spec_noise, ref=np.max)

    spec_reverb = np.abs(librosa.stft(x_reverb, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec_reverb =  librosa.amplitude_to_db(spec_reverb, ref=np.max)

    # Get the required frequency bins
    spec = spec[:nfreq]
    spec_noise = spec_noise[:nfreq]
    spec_reverb = spec_reverb[:nfreq]
    
    # Arrange into (C, F, T), C = 3
    spec = np.expand_dims(spec,0)
    spec_noise = np.expand_dims(spec_noise,0)
    spec_reverb = np.expand_dims(spec_reverb,0)
    spec = np.concatenate((spec, spec_noise, spec_reverb), axis=0)

    return spec



def extract_logmelspec(x, sr, params):
 
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    n_mels        = params['nmel']
    

    #calculate stft
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels,
                                        n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window)
    
    logmelspec =  librosa.power_to_db(melspec, ref=np.max)

    # Expand to (C, F, T), C = 3
    logmelspec =  np.expand_dims(logmelspec, 0)
    
    return logmelspec


def extract_logmeldeltaspec(x, sr, params):
      
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    n_mels        = params['nmel']
    
    #calculate stft
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels,
                                        n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window)
    logmelspec =  librosa.power_to_db(melspec, ref=np.max)
    
    #logmeldeltaspec = librosa.feature.delta(logmelspec)
    #logmeldelta2spec = librosa.feature.delta(logmelspec, order=2)
    logmeldeltaspec = librosa.feature.delta(logmelspec,width=5,mode='nearest')
    logmeldelta2spec = librosa.feature.delta(logmeldeltaspec, width=5,mode='nearest')
    
    #Arrange into (C, F, T), C = 3
    logmelspec = np.expand_dims(logmelspec, axis=0)
    logmeldeltaspec = np.expand_dims(logmeldeltaspec, axis=0)
    logmeldelta2spec = np.expand_dims(logmeldelta2spec, axis=0)
    logmelspec = np.concatenate((logmelspec, logmeldeltaspec, logmeldelta2spec), axis=0)
    
    return logmelspec


def segment_nd_features(data, emotion, segment_size):
    '''
    Segment features into <segment_size> frames.
    Pad with 0 if data frames < segment_size

    Input:
    ------
        - data: shape is (Channels, Fime, Time)
        - emotion: emotion label for the current utterance data
        - segment_size: length of each segment
    
    Return:
    -------
    Tuples of (number of segments, frames, segment labels, utterance label)
        - frames: ndarray of shape (N, C, F, T)
                    - N: number of segments
                    - C: number of channels
                    - F: frequency index
                    - T: time index
        - segment labels: list of labels for each segments
                    - len(segment labels) == number of segments
    '''
    # Transpose data to C, T, F
    data = data.transpose(0,2,1)

    time = data.shape[1]
    nch = data.shape[0]
    start, end = 0, segment_size
    num_segs = math.ceil(time / segment_size) # number of segments of each utterance
    data_tot = []

    for i in range(num_segs):
        # The last segment
        if end > time:
            end = time
            start = max(0, end - segment_size)
        
        # Do padding
        data_pad = []
        for c in range(nch):
            data_ch = data[c]
            data_ch = np.pad(
                data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
            data_pad.append(data_ch)

        data_pad = np.array(data_pad)
        
        # Stack
        data_tot.append(data_pad)
        
        # Update variables
        start = end
        end = min(time, end + segment_size)

    data_tot = np.stack(data_tot)
    utt_label = emotion
    segment_labels = [emotion] * num_segs
    
    #Transpose output to N,C,F,T
    data_tot = data_tot.transpose(0,1,3,2)

    return (num_segs, data_tot, segment_labels, utt_label)

#Feature extraction function map
GET_FEATURES = {'logspec': extract_logspec,
                'logmelspec': extract_logmelspec,
                'logmeldeltaspec': extract_logmeldeltaspec,
                'lognrevspec': extract_lognrevspec
                }