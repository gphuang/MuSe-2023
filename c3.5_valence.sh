#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c5_1st_step_%A.out
#SBATCH --job-name=muse_c5_1
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

### VALENCE

# BPM 27837980
# ECG 27837981
# resp 27837982

for model_dim in 32 128 256; do
for rnn_n_layers in 2 4 8; do
for lr in 0.001 0.002 0.005 0.01; do 
for win_len in 50 100 200 250; do 
for hop_len in 25 50 100; do 
for rnn_dropout in 0.1 0.2 0.5 0.9; do 
python3 main.py --task personalisation --feature resp \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu
python3 main.py --task personalisation --feature resp \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 --rnn_bi\
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu                   
done 
done
done
done 
done
done