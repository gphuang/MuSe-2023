#!/bin/bash
#SBATCH --time=47:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env

for emo_dim in physio-arousal valence
do 
for model_dim in 128 64 32
do 
for rnn_n_layers in 3 2 1
do 
for lr in 0.001
do 
for win_len in 100
do 
hop_len=10 # $((win_len/2))
echo $emo_dim $model_dim $rnn_n_layers $lr  $win_len $hop_len 
/usr/bin/time -v python3 main.py --task personalisation --feature mfcc_ecg --normalize \
            --emo_dim $emo_dim --model_dim $model_dim --rnn_n_layers $rnn_n_layers \
            --lr $lr --win_len $win_len  --hop_len $hop_len \
            --model_type RNN --rnn_bi --rnn_dropout 0.5 --use_gpu
done
done
done
done
done

## pilot
# ecg-valence 0.1029
# RNN_2024-04-18-18-30 egemaps_ecg 0.2886 
# RNN_2024-04-19-11-28 mfcc_ecg lr 0.001 0.3327
# RNN_2024-04-17-18-49 default 0.3183
# RNN_2024-04-19-12-41 dim 256 0.2753
# RNN_2024-04-19-13-14 lr 0.0001 0.1965
# RNN_2024-04-19-13-12 win 50 0.2869
# RNN_2024-04-19-13-13 hop 10 0.4411 *** 
# ecg-arousal mfcc_ecg 0.0071

## mfcc_ecg
# 30971589 lr [0.005, 0.001, 0.0005] win_len [200 100 50] hop_len=win_len/2
# 30971592 model_dim [128 64 32] rnn_n_layers [3 2 1] lr=0.001 win_len=100 hop_len=10

# feats?_ecg
# https://neuropsychology.github.io/NeuroKit/functions/ecg.html#analysis

# resp-valence 
## RNN_2024-04-18-15-20 resp 0.2278 
## RNN_2024-04-18-15-27 mfcc_resp 0.1465

## rnn vs. crnn add conv layer between linear and rnn layers, no pooling
                
