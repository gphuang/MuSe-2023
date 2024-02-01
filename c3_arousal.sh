#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3.5_arousal_%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda

source activate muse

### fine-tune RNN on biosignals for AROUSAL & VALENCE. 

# optimise the GRU’s hidden representations’ size 
# the number of stacked GRU layers 
# the learning rate.
# consider both unidirectional and bidirectional GRUs.
# Window size and step length of the segmentation
# rnn_dropout

### AROUSAL

# BPM
python3 main.py --task personalisation --feature BPM \
                    --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.01 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# ECG
python3 main.py --task personalisation --feature ECG \
                    --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.01 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu
# resp
python3 main.py --task personalisation --feature resp \
                    --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.01 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu
