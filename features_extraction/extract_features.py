import os
import sys
import argparse
import numpy as np
import pickle
from features_util import get_utterance_files, extract_data
from collections import Counter
import pandas as pd


def main(args):
    # Map emotions with integer labels
    #emot_map = {"ang": 0, "sad": 1, "hap": 2, "exc": 2, "neu": 3}
    # Emotion classes to be extracted
    emot_map = {"ang": 0, "sad": 1, "hap": 2, "neu": 3}
    
    #Get spectrogram parameters
    params={'window'        : args.window,
            'win_length'    : args.win_length,
            'hop_length'    : args.hop_length,
            'ndft'          : args.ndft,
            'nfreq'         : args.nfreq,
            'nmel'          : args.nmel,
            'segment_size'  : args.segment_size
            }
    
    dataset  = args.dataset
    features = args.features
    dataset_dir = args.dataset_dir
    if args.save_dir is not None:
        out_filename = args.save_dir+dataset+'_'+args.save_label+'.pkl'
    else:
        out_filename = 'None'

    print('\n')
    print('*'*50)
    print('\nFEATURES EXTRACTION')
    print(f'\t{"Dataset":>15}: {dataset}')
    print(f'\t{"Features":>15}: {features}')
    print(f'\t{"Dataset dir.":>15}: {dataset_dir}')
    print(f'\t{"Features file":>15}: {out_filename}')
    print(f"\nPARAMETERS:")
    for key in params:
        print(f'\t{key:>15}: {params[key]}')
    print('\n')

    #get utterance and label filenames, organized into dictionary
    #   {speaker_id:([utterance_wav_filenames], [label_filenames])}
    speaker_files = get_utterance_files(dataset, dataset_dir)
    
    #extract features to dictionary 
    #   {speakerID: (data_tot, labels_tot, labels_segs_tot, segs)}
    features_data = extract_data(dataset, dataset_dir, features, emot_map, speaker_files, params)
   
    #save features
    if args.save_dir is not None:
        
        with open(out_filename, "wb") as fout:
                pickle.dump(features_data, fout)

    #Print classes statistic
        
    print(f'\nSEGMENT CLASS DISTRIBUTION PER SPEAKER:\n')
    
    class_dist= []
    speakers=[]
    data_shape=[]
    for speaker in features_data.keys():
        #print(f'\tSpeaker {speaker:>2}: {sorted(Counter(features_data[speaker][2]).items())}')
        cnt = sorted(Counter(features_data[speaker][2]).items())
        class_dist.append([x[1] for x in cnt])
        speakers.append(speaker)
        data_shape.append(str(features_data[speaker][0].shape))
    class_dist = np.array(class_dist)
    
    class_dist_f = pd.DataFrame({"speaker": speakers,
                                 "shape (N,C,F,T)": data_shape,
                                 "ang": class_dist[:,0],
                                 "sad": class_dist[:,1],
                                 "hap": class_dist[:,2],
                                 "neu": class_dist[:,3]})
    class_dist_f = class_dist_f.to_string(index=False) 
    print(class_dist_f)
     
    print('\n')
    print('*'*50)
    print('\n')



def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #DATASET
    parser.add_argument('dataset', type=str, default='IEMOCAP',
        help='Dataset to extract features. Options:'
             '  - IEMOCAP (default)'
             '  - emoDB')
    parser.add_argument('dataset_dir', type=str,
        help='Path to the dataset directory.')
    
    #FEATURES
    parser.add_argument('--features', type=str, default='logspec',
        help='Feature to be extracted. Options:'
             '  - logspec (default) : (1 ch.)log spectrogram'
             '  - logmelspec        : (1 ch.)log mel spectrogram'
             '  - logmeldeltaspec   : (3 ch.)log mel spectrogram, delta, delta-delta'
             '  - lognrevspec       : (3 ch.)log spectrograms of signal, signal+noise, signal+reverb')
    
    parser.add_argument('--window', type=str, default='hamming',
        help='Window type. Default: hamming')

    parser.add_argument('--win_length', type=float, default=40,
        help='Window size (msec). Default: 40')

    parser.add_argument('--hop_length', type=float, default=10,
        help='Window hop size (msec). Default: 10')
    
    parser.add_argument('--ndft', type=int, default=800,
        help='DFT size. Default: 800')

    parser.add_argument('--nfreq', type=int, default=200,
        help='Number of lowest DFT points to be used as features. Default: 200'
             '  Only effective for <logspec, lognrevspec> features')
    
    parser.add_argument('--nmel', type=int, default=128,
        help='Number of mel frequency bands used as features. Default: 128'
             '  Only effectice for <logmel, logmeldeltaspec> features')
    
    parser.add_argument('--segment_size', type=int, default=300,
        help='Size of each features segment')


    #FEATURES FILE
    parser.add_argument('--save_dir', type=str, default=None,
        help='Path to directory to save the extracted features.')
    
    parser.add_argument('--save_label', type=str, default=None,
        help='Label to save the feature')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))