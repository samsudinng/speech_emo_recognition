import pickle
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn import preprocessing
from collections import Counter
import albumentations as A
import random
from tqdm import tqdm

SCALER_TYPE = {'standard':preprocessing.StandardScaler,
               'minmax'  :preprocessing.MinMaxScaler
              }


class TrainLoader(torch.utils.data.Dataset):
    """
    Holds data for a train dataset (e.g., holds training examples as well as
    training labels.)

    Parameters
    ----------
    data : ndarray
        Input data.
    target : ndarray
        Target labels.
    

    """
    def __init__(self, data, target, num_classes=7, pre_process=None):
        super(TrainLoader).__init__()
        self.data = data
        self.target = target
        self.n_samples = data.shape[0]
        self.num_classes = num_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def weighted_accuracy(self, predictions, target):
        """
        Calculate accuracy score given the predictions.

        Parameters
        ----------
        predictions : ndarray
            Model's predictions.

        Returns
        -------
        float
            Accuracy score.

        """
        acc = (target == predictions).sum() / self.n_samples
        return acc


    def unweighted_accuracy(self, predictions,target):
        """
        Calculate unweighted accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Unweighted Accuracy (UA) score.

        """


        class_acc = 0
        n_classes = 0
        for c in range(self.num_classes):
            class_pred = np.multiply((target == predictions),
                                     (target == c)).sum()
            
            if (target == c).sum() > 0:
                 class_pred /= (target == c).sum()
                 n_classes += 1

                 class_acc += class_pred
            
        return class_acc / n_classes



class TestLoader(torch.utils.data.Dataset):
    """
    Holds data for a validation/test set.

    Parameters
    ----------
    data : ndarray
        Input data of shape `N x H x W x C`, where `N` is the number of examples
        (segments), `H` is image height, `W` is image width and `C` is the
        number of channels.
    actual_target : ndarray
        Actual target labels (labels for utterances) of shape `(N',)`, where
        `N'` is the number of utterances.
    seg_target : ndarray
        Labels for segments (note that one utterance might contain more than
        one segments) of shape `(N,)`.
    num_segs : ndarray
        Array of shape `(N',)` indicating how many segments each utterance
        contains.
    num_classes :
        Number of classes.
    """

    def __init__(self, data, actual_target, seg_target,
                 num_segs, num_classes=7):
        super(TestLoader).__init__()
        self.data = data
        self.target = seg_target
        self.n_samples = data.shape[0]
        self.n_actual_samples = actual_target.shape[0]

        self.actual_target = actual_target
        self.num_segs = num_segs
        self.num_classes = num_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def get_preds(self, seg_preds):
        """
        Get predictions for all utterances from their segments' prediction.
        This function will accumulate the predictions for each utterance by
        taking the maximum probability along the dimension 0 of all segments
        belonging to that particular utterance.
        """
        preds = np.empty(
            shape=(self.n_actual_samples, self.num_classes), dtype="float")

        end = 0
        for v in range(self.n_actual_samples):
            start = end
            end = start + self.num_segs[v]
            
            preds[v] = np.max(seg_preds[start:end], axis=0)
            
        preds = np.argmax(preds, axis=1)
        return preds


    def weighted_accuracy(self, utt_preds):
        """
        Calculate accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Accuracy score.

        """

        acc = (self.actual_target == utt_preds).sum() / self.n_actual_samples
        return acc


    def unweighted_accuracy(self, utt_preds):
        """
        Calculate unweighted accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Unweighted Accuracy (UA) score.

        """
        class_acc = 0
        n_classes = 0
        
        for c in range(self.num_classes):
            class_pred = np.multiply((self.actual_target == utt_preds),
                                     (self.actual_target == c)).sum()

        
            if (self.actual_target == c).sum() > 0:    
                class_pred /= (self.actual_target == c).sum()
                n_classes += 1
                class_acc += class_pred

        return class_acc / n_classes


    def confusion_matrix(self, utt_preds):
        """Compute confusion matrix given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        """
        conf = confusion_matrix(self.actual_target, utt_preds)
        
        # Make confusion matrix into data frame for readability
        conf_fmt = pd.DataFrame({"ang": conf[:, 0], "bor": conf[:, 1],
                             "dis": conf[:, 2], "fear": conf[:, 3],
                             "hap": conf[:, 4], "sad": conf[:, 5],
                             "neu": conf[:, 6]})
        conf_fmt = conf_fmt.to_string(index=False)
        return (conf, conf_fmt)

    
    def confusion_matrix_iemocap(self, utt_preds):
        """Compute confusion matrix given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        """
        conf = confusion_matrix(self.actual_target, utt_preds)
        
        # Make confusion matrix into data frame for readability
        conf_fmt = pd.DataFrame({"ang": conf[:, 0], "sad": conf[:, 1],
                             "hap": conf[:, 2], "neu": conf[:, 3]})
        conf_fmt = conf_fmt.to_string(index=False)
        return (conf, conf_fmt)


class DatasetLoader:
    """
    Wrapper for both `TrainLoader` and `TestLoader`, which loads pre-processed
    speech features into `Dataset` objects.

    Parameters
    ----------
    data : tuple
        Data extracted using `extract_features.py`.
    num_classes : int
        Number of classes.
    pre_process : fn
        A function to be applied to `data`.

    """
    def __init__(self, features_data,
                val_speaker_id='1M', test_speaker_id='1F', 
                scaling='standard', oversample=False,
                augment=False):
        
        #features_data format: dictionary
        #    {speaker_id: (data_tot, labels_tot, labels_segs_tot, segs)}
        #data shape (N_segment, Channels, Freq., Time)
        

        #get training dataset
        train_data, train_labels = None, None
        for speaker_id in features_data.keys():
            if speaker_id in [val_speaker_id, test_speaker_id]:
                continue
            
            #Concatenate all training features segment
            if train_data is None:
                train_data = features_data[speaker_id][0].astype(np.float32)
            else:
                train_data = np.concatenate((train_data, 
                                            features_data[speaker_id][0].astype(np.float32) ),
                                            axis=0)
            
            #Concatenate all training segment labels
            if train_labels is None:
                train_labels = features_data[speaker_id][2].astype(np.long)
            else:
                train_labels = np.concatenate((train_labels,
                                               features_data[speaker_id][2].astype(np.long)),
                                               axis=0)
        
        if augment == True:
            #perform training data augmentation
            self.train_data, self.train_labels = data_augment(train_data, train_labels)
        else:
            self.train_data = train_data
            self.train_labels = train_labels
        self.num_classes=len(Counter(train_labels).items())
        self.num_in_ch = self.train_data.shape[1]

        #get validation dataset
        self.val_data       = features_data[val_speaker_id][0].astype(np.float32)
        self.val_seg_labels = features_data[val_speaker_id][2].astype(np.long)
        self.val_labels     = features_data[val_speaker_id][1].astype(np.long)
        self.val_num_segs   = features_data[val_speaker_id][3]

        #get validation dataset
        self.test_data       = features_data[test_speaker_id][0].astype(np.float32)
        self.test_seg_labels = features_data[test_speaker_id][2].astype(np.long)
        self.test_labels     = features_data[test_speaker_id][1].astype(np.long)
        self.test_num_segs   = features_data[test_speaker_id][3]

        #Normalize dataset
        self._normalize(scaling)
    
        #random oversampling on training dataset
        if oversample == True:
            print('\nPerform training dataset oversampling')
            datar, labelr = random_oversample(self.train_data, self.train_labels)
            self.train_data = datar
            self.train_labels = labelr
        

        assert self.val_data.shape[0] == self.val_seg_labels.shape[0] == sum(self.val_num_segs)
        assert self.val_labels.shape[0] == self.val_num_segs.shape[0]
        assert self.test_data.shape[0] == self.test_seg_labels.shape[0] == sum(self.test_num_segs)
        assert self.test_labels.shape[0] == self.test_num_segs.shape[0]
            
        print('\n<<DATASET>>\n')
        print(f'Val. speaker id : {val_speaker_id}')
        print(f'Test speaker id : {test_speaker_id}')
        print(f'Train data      : {self.train_data.shape}')
        print(f'Train labels    : {self.train_labels.shape}')
        print(f'Eval. data      : {self.val_data.shape}')
        print(f'Eval. label     : {self.val_labels.shape}')
        print(f'Eval. seg labels: {self.val_seg_labels.shape}')
        print(f'Eval. num seg   : {self.val_num_segs.shape}')
        print(f'Test data       : {self.test_data.shape}')
        print(f'Test label      : {self.test_labels.shape}')
        print(f'Test seg labels : {self.test_seg_labels.shape}')
        print(f'Test num seg    : {self.test_num_segs.shape}')
        print('\n')

    
    def _normalize(self, scaling):
        '''
        calculate normalization factor from training dataset and apply to
           the whole dataset
        '''
        #get data range
        input_range = self._get_data_range()

        #re-arrange array from (N, C, F, T) to (C, -1, F)
        nsegs = self.train_data.shape[0]
        nch   = self.train_data.shape[1]
        nfreq = self.train_data.shape[2]
        ntime = self.train_data.shape[3]
        rearrange = lambda x: x.transpose(1,0,3,2).reshape(nch,-1,nfreq)
        self.train_data = rearrange(self.train_data)
        self.val_data   = rearrange(self.val_data)
        self.test_data  = rearrange(self.test_data)
        
        #scaler type
        scaler = SCALER_TYPE[scaling]()

        for ch in range(nch):
            #get scaling values from training data
            scale_values = scaler.fit(self.train_data[ch])
            
            #apply to all
            self.train_data[ch] = scaler.transform(self.train_data[ch])
            self.val_data[ch] = scaler.transform(self.val_data[ch])
            self.test_data[ch] = scaler.transform(self.test_data[ch])
        
        #Shape the data back to (N,C,F,T)
        rearrange = lambda x: x.reshape(nch,-1,ntime,nfreq).transpose(1,0,3,2)
        self.train_data = rearrange(self.train_data)
        self.val_data   = rearrange(self.val_data)
        self.test_data  = rearrange(self.test_data)

        print(f'\nDataset normalized with {scaling} scaler')
        print(f'\tRange before normalization: {input_range}')
        print(f'\tRange after  normalization: {self._get_data_range()}')

    def _get_data_range(self):
        #get data range
        trmin = np.min(self.train_data)
        evmin = np.min(self.val_data)
        tsmin = np.min(self.test_data)
        dmin = np.min(np.array([trmin, evmin, tsmin]))

        trmax = np.max(self.train_data)
        evmax = np.max(self.val_data)
        tsmax = np.max(self.test_data)
        dmax = np.max(np.array([trmax, evmax, tsmax]))
        
        return [dmin, dmax]


    def get_train_dataset(self):
        return TrainLoader(
            self.train_data, self.train_labels, num_classes=self.num_classes)
    
    def get_val_dataset(self):
        return TestLoader(
            self.val_data, self.val_labels,
            self.val_seg_labels, self.val_num_segs,
            num_classes=self.num_classes)
    
    def get_test_dataset(self):
        return TestLoader(
            self.test_data, self.test_labels,
            self.test_seg_labels, self.test_num_segs,
            num_classes=self.num_classes)


def random_oversample(data, labels):
    print('\tOversampling method: Random Oversampling')
    ros = RandomOverSampler(random_state=0)

    n_samples = data.shape[0]
    fh = data.shape[2]
    fw = data.shape[3]
    n_features= fh*fw
        
    data = np.squeeze(data,axis=1)
    data = np.reshape(data,(n_samples, n_features))
    data_resampled, label_resampled = ros.fit_resample(data, labels)
    n_samples = data_resampled.shape[0]
    data_resampled = np.reshape(data_resampled,(n_samples,fh,fw))
    data_resampled = np.expand_dims(data_resampled, axis=1)
    
    return data_resampled, label_resampled


def data_augment(data, label):
    """
    Perform data augmentation based on albumentations package.
    Transform pipeline:
        - Time masking
    """
    aug_data, aug_label = None, None
    data_and_label = zip(data, label)
    #for i in tqdm(range(data.shape[0]), desc='Data Augmentation'):
    for spec, spec_label in tqdm(data_and_label, desc='Data Augmentation'):
        #spec = data[i].copy().squeeze(axis=0)
        spec = spec.squeeze(axis=0)
        spec_aug = spec_augment(spec, num_mask=1, time_masking_max_percentage=0.25 )
        spec_aug = np.expand_dims(spec_aug, axis=0)
        if aug_data is None:
            aug_data = spec_aug
            aug_label= spec_label
        else:
            aug_data = np.vstack((aug_data, spec_aug))
            aug_label = np.append(aug_label, spec_label)
    
    #append augmented data to training data
    aug_data = np.expand_dims(aug_data, axis=1)
    data = np.vstack((data,aug_data))
    label= np.append(label,aug_label)

    return data, label

def spec_augment(spec: np.ndarray, num_mask=0, 
                 freq_masking_max_percentage=0.0, time_masking_max_percentage=0.0):

    """
    Quick implementation of Google's SpecAug
    Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
    """
    
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec

