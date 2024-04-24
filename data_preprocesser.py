import os
import sys
import librosa
import opensmile
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import librosa
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

def parse_args():

    parser = argparse.ArgumentParser(description='Preprocess data: feature extraction.')
    parser.add_argument('--feat_extractor', type=str, default='mfcc', choices=['melspec', 'mfcc', 'egemaps', 'hubert'],
                        help=f'Specify the feat.')
    
    args = parser.parse_args()
    return args

def segment_ndarray(sample:np.ndarray, win_len, hop_len, padding=False):
    """
    source: https://github.com/EIHW/MuSe-2023/blob/bb5f7f2332192e5a3f603728c36385bd4a3000c7/data_parser.py
    tbd: pad_mode in librosay mfcc 'center=True'. 1 extra step
    tbd: pad_mode = edge in egamaps. missing 1~2 steps
    tbd: spkr_train _devel _test merge, extract, and re-split?
    """
    segmented_sample = []
    if len(sample) > win_len and padding:
        sample = np.pad(sample, (win_len//2, win_len//2), 'edge')

    for s_idx in range(0, len(sample), hop_len):
        e_idx = min(s_idx + win_len, len(sample))
        segment = sample[s_idx:e_idx]
        segmented_sample.append(segment)
        if e_idx == len(sample):
            break

    return segmented_sample

def main(args):
    df_partition = pd.read_csv('/scratch/elec/puhe/c/muse_2023/c3_muse_personalisation/metadata/partition.csv')
    dir_in = '/scratch/elec/puhe/c/muse_2023/c3_muse_personalisation/raw_data'
    dir_out = '/scratch/elec/puhe/c/muse_2023/c3_muse_personalisation/feature_segments'
    _signals = ['wav'] #  ['BPM', 'ECG', 'resp'] 
    _ext = '.wav' # '.csv'
    feat_extractor = args.feat_extractor # ['melspec', 'mfcc', 'egemaps']
    hop_len = 500 # input 1kHz output 2Hz
    win_len = 1000 
    fnames = df_partition['Id'].to_list() 
    # print(len(fnames)) 97
    if feat_extractor == 'hubert':
        model_name = "superb/hubert-large-superb-er"  
        model = HubertForSequenceClassification.from_pretrained(model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    for _signal in _signals:
        print(f'Processing {_signal}')
        _dir_out = os.path.join(dir_out, feat_extractor + '-' + _signal.lower()) # MUSE uses '-' in feat name string. 
        if not os.path.exists(_dir_out):
            os.makedirs(_dir_out)
        for _fname in ['10', '54_train']:# fnames:
            print(f'Speaker id: {_fname}')
            subject_id = _fname.split('_')[0]
            out_fname = os.path.join(_dir_out, _fname + '.csv')
            in_fname = os.path.join(dir_in, _signal, _fname + _ext)
            if _ext == '.csv':
                sampling_rate = 1000 # biosignal input
                df = pd.read_csv(in_fname)            
                data = df[_signal].to_numpy()
            if _ext == '.wav':
                data, sampling_rate = librosa.load(in_fname, sr=16000) # specific for hubert pretrained model
            if feat_extractor == 'hubert':
                # slicing
                samples = segment_ndarray(data, win_len=sampling_rate, hop_len=sampling_rate//2)
                _feats = []
                for i, segment in enumerate(samples):
                    inputs = feature_extractor(segment, sampling_rate=sampling_rate, padding=True, return_tensors="pt")
                    inputs = {key: inputs[key].to(device).float() for key in inputs}
                    logits = model(**inputs).logits # (n_emo_class,)
                    _feats.append(logits.detach().numpy().squeeze())
                feats = np.asarray(_feats) # (seq_len, n_emo_class)
                #print(feats.shape) 
                # sys.exit(0)

            if feat_extractor == 'melspec':
                melspecs = librosa.feature.melspectrogram(y=data, sr=sampling_rate, n_mels=128, hop_length=sampling_rate//2) # (128, seq_len)
                feats = melspecs.T # (seq_len, n_mel)
            if feat_extractor == 'mfcc':
                mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=20, hop_length=sampling_rate//2) # (20, seq_len)
                feats = mfccs.T # (seq_len, n_mfcc)
            if feat_extractor == 'egemaps':
                smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,)
                # slicing
                samples = segment_ndarray(data, win_len=sampling_rate, hop_len=sampling_rate//2, padding=True)
                _feats = []
                for i, segment in enumerate(samples):
                    egemaps = smile.process_signal(segment, sampling_rate) # (n_egemaps,)
                    _feats.append(egemaps.values.squeeze()) 
                feats = np.asarray(_feats)  # (seq_len, n_egemaps)
            if feats.shape[0] == 1:
                feats = np.tile(feats,(df.shape[0], 1))
            else:
                feats = feats[:-1, :] # remove one window_len e.g. padding from mfcc
            col_values = [str(i) for i in range(feats.shape[1])] 
            timestamp = [hop_len*i for i in range(feats.shape[0])] 
            df = pd.DataFrame(data = feats, columns = col_values)
            df['timestamp'] = timestamp
            df['subject_id'] = subject_id
            cols = ['timestamp', 'subject_id',] + col_values
            df = df[cols]
            df.to_csv(out_fname, index=False)

            """print(df.shape)
            print(df.head(3))
            _df = pd.read_csv(out_fname)
            print(_df.head(3))
            sys.exit(0)"""
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
