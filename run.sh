#!/bin/bash
#SBATCH --time=48:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env

# feature='ECG' #'melspec-ecg' #'mfcc-ecg' #'egemaps-ecg' #    
# for feature in resp melspec-resp mfcc-resp egemaps-resp 
for feature in BPM melspec-bpm mfcc-bpm egemaps-bpm
# for feature in melspec-wav mfcc-wav # 
# for feature in hubert-wav 
do 
for emo_dim in valence # physio-arousal 
do 
for model_dim in 512 256 128
do 
for rnn_n_layers in 4 2
do 
for lr in 0.005 0.001
do 
for win_len in 200 100 50
do 
for hop_len in 25 10
do
echo $feature $emo_dim $model_dim $rnn_n_layers $lr  $win_len $hop_len 
/usr/bin/time -v python3 main.py --task personalisation --feature $feature --normalize \
            --emo_dim $emo_dim --model_dim $model_dim --rnn_n_layers $rnn_n_layers \
            --lr $lr --win_len $win_len  --hop_len $hop_len \
            --model_type RNN --rnn_bi --rnn_dropout 0.5 --use_gpu
done
done
done
done
done
done
done

## ECG 
# pilot
# ecg-valence 0.1029
# RNN_2024-04-18-18-30 egemaps_ecg 0.2886 
# RNN_2024-04-19-11-28 mfcc_ecg lr 0.001 0.3327
# RNN_2024-04-17-18-49 default 0.3183
# RNN_2024-04-19-12-41 dim 256 0.2753
# RNN_2024-04-19-13-14 lr 0.0001 0.1965
# RNN_2024-04-19-13-12 win 50 0.2869
# RNN_2024-04-19-13-13 hop 10 0.4411  
# ecg-arousal mfcc_ecg 0.0071
# RNN_2024-04-[19, 20] mfcc-ecg arousal 0.2923, [128_2_True_64]_[0.005_256]
# RNN_2024-04-[19, 20] mfcc-ecg valence 0.4411, [128_2_True_64]_[0.001_256]
# RNN_2024-04-20-13-31 mfcc-ecg arousal 0.3487, [128_3_True_64]_[0.005_256]
# RNN_2024-04-20-09-21 mfcc-ecg valence 0.4199, [128_3_True_64]_[0.001_256]
# RNN_2024-04-20-11-59 egemaps-ecg valence 0.4545 [128_4_True_64]_[0.005_256]
# loop 50hrs 10mins per loop
# RNN_2024-04-22-12-26_[mfcc-ecg]_[physio-arousal]_[128_3_True_64]_[0.005_256]    0.3966
# RNN_2024-04-22-16-14_[mfcc-ecg]_[valence]_[256_3_True_64]_[0.005_256]           0.4338 
# RNN_2024-04-22-09-45_[egemaps-ecg]_[physio-arousal]_[256_3_True_64]_[0.005_256] 0.4821    
# RNN_2024-04-22-18-18_[egemaps-ecg]_[valence]_[128_3_True_64]_[0.005_256]        0.3869
# RNN_2024-04-24-01-25_[melspec-ecg]_[physio-arousal]_[128_2_True_64]_[0.005_256] ID/Seed 103 0.5574
# RNN_2024-04-24-02-02_[mfcc-ecg]_[valence]_[256_4_True_64]_[0.001_256]           ID/Seed 105 0.5395

## BPM
# RNN_2024-04-24-06-59_[melspec-bpm]_[physio-arousal]_[512_4_True_64]_[0.001_256] ID/Seed 101 0.4749
# RNN_2024-04-27-00-41_[melspec-bpm]_[valence]_[128_4_True_64]_[0.005_256]        ID/Seed 104 0.4883

## resp
# RNN_2024-04-23-09-04_[egemaps-resp]_[physio-arousal]_[512_4_True_64]_[0.005_256] ID/Seed 105 0.4438
# RNN_2024-04-24-21-13_[melspec-resp]_[valence]_[256_4_True_64]_[0.005_256]        ID/Seed 105 0.3990

## hubert-large-superb-er
# RNN_2024-04-25-18-16_[hubert-wav]_[physio-arousal]_[512_2_True_64]_[0.005_256] ID/Seed 104 0.4250
# RNN_2024-04-26-00-37_[hubert-wav]_[valence]_[512_2_True_64]_[0.001_256]        ID/Seed 102 0.5333 

## update c3_affect_1st_step.sh - rename self.rnn back to self.encoder, train best model on c3
# RNN_2024-04-29 (verify if 'best' results were reproducible.) 31188809 
 
## personalize (skipped: params. optimization on 60s data.)
# arousal win_len 20 hop 10, lr 0.002
# valence win_len 10 hop 5, lr 0.002

## predict
# RNN_2024-04-24-02-02_[mfcc-ecg]_[valence]_[256_4_True_64]_[0.001_256] ID/Seed 105 0.5395 
# RNN_2024-04-24-01-25_[melspec-ecg]_[physio-arousal]_[128_2_True_64]_[0.005_256] ID/Seed 103 0.5574

## fusion?
# naive: feature, decision, average across
# intermedia: siamese

## memory & attention € pretraining
# rnn vs. crnn add conv layer between linear and rnn layers

## Notes
# wav feature extraction with e.g. mfcc, egemaps, mismatch 1 hop
