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

# BPM
python3 main.py --task personalisation --feature BPM \
                    --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# ECG                
python3 main.py --task personalisation --feature ECG \
                    --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# resp
python3 main.py --task personalisation --feature resp \
                    --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu