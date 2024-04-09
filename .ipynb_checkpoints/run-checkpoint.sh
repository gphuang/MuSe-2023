#!/bin/bash
#SBATCH --time=05:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env


python3 main.py --task personalisation --feature ECG --normalize \
            --emo_dim physio-arousal --model_dim 32 --rnn_n_layers 2 --rnn_bi --lr 0.05 \
            --win_len 200 --hop_len 100 --rnn_dropout 0.5 --use_gpu

python3 main.py --task personalisation --feature ECG \
            --emo_dim physio-arousal --model_dim 128 --lr 0.005 \
            --model_type FNN --win_len 200 --hop_len 100 --use_gpu
            
python3 main.py --task personalisation --feature ECG \
            --emo_dim physio-arousal --model_dim 32 --rnn_n_layers 4 --rnn_bi --lr 0.05 \
            --model_type CRNN --win_len 200 --hop_len 100 --rnn_dropout 0.5 --use_gpu
            
## rnn vs. crnn add conv layer between linear and rnn layers, no pooling
# egemaps 30682441 
# ds 30685538 

# resp
# https://arxiv.org/abs/2103.02183

                
