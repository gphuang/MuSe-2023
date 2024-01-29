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

config = {
    "model_dim": tune.choice([32*2**i for i in range(5)]),
    "rnn_n_layers": tune.choice([2, 4, 8]),
    "rnn_bi": tune.choice([True, False]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "win_len": tune.choice([50*i for i in range(1, 5)]),
    "hop_len": tune.choice([25*i for i in range(1, 5)]),
    "rnn_dropout": tune.choice([0.1*i for i in range(10)]),

### AROUSAL

# BPM
python3 main.py --task personalisation --feature BPM \
                    --normalize --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 --rnn_bi\
                    --lr 0.002 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# ECG
python3 main.py --task personalisation --feature ECG \
                    --normalize --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 --rnn_bi\
                    --lr 0.002 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu
# resp
python3 main.py --task personalisation --feature resp \
                    --normalize --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 --rnn_bi\
                    --lr 0.002 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu                   

### VALENCE

# BPM
python3 main.py --task personalisation --feature BPM \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 --rnn_bi\
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# ECG                
python3 main.py --task personalisation --feature ECG \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 --rnn_bi\
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# resp
python3 main.py --task personalisation --feature resp \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 --rnn_bi\
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu

