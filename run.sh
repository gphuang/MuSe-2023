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

python data_preprocesser.py --feat_extractor melspec

feature='melspec-resp' # 'melspec-ecg' # 'egemaps-resp' # 'mfcc-resp' # 'mfcc-ecg' # 'egemaps-ecg' # 
for emo_dim in physio-arousal valence
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
echo $emo_dim $model_dim $rnn_n_layers $lr  $win_len $hop_len 
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
# loop 6hrs
# RNN_2024-04-[19, 20] mfcc-ecg arousal 0.2923, [128_2_True_64]_[0.005_256]
# RNN_2024-04-[19, 20] mfcc-ecg valence 0.4411, [128_2_True_64]_[0.001_256]
# RNN_2024-04-20-13-31 mfcc-ecg arousal 0.3487, [128_3_True_64]_[0.005_256]
# RNN_2024-04-20-09-21 mfcc-ecg valence 0.4199, [128_3_True_64]_[0.001_256]
# RNN_2024-04-20-11-59 egemaps-ecg valence 0.4545 [128_4_True_64]_[0.005_256]
# loop 50hrs 10mins per loop
# RNN_2024-04-22-12-26_[mfcc-ecg]_[physio-arousal]_[128_3_True_64]_[0.005_256]    0.3966
# RNN_2024-04-22-16-14_[mfcc-ecg]_[valence]_[256_3_True_64]_[0.005_256]           win_len 200 hop_len 10 0.4338 ***
# RNN_2024-04-22-09-45_[egemaps-ecg]_[physio-arousal]_[256_3_True_64]_[0.005_256] win_len 50  hop_len 25 0.4821 ***  
# RNN_2024-04-22-18-18_[egemaps-ecg]_[valence]_[128_3_True_64]_[0.005_256]        0.3869
# loop 48hrs +melspec
# RNN_2024-04-23 31027956 31027955 31040231
# c3_ecg eval

## RESP
# pilot 
# RNN_2024-04-18-15-20 resp valence 0.2278 
# RNN_2024-04-18-15-27 mfcc_resp valence 0.1465
# loop 48hrs +melspec
# RNN_2024-04-22 31027958 31027957 31040236

## feature fusion
# naive
# siamese

## rnn vs. crnn add conv layer between linear and rnn layers, no pooling
