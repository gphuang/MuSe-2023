#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3_bio_%A.out
#SBATCH --job-name=muse_c3_bio
#SBATCH -n 1

source activate data2vec

### biosignals for AROUSAL & VALENCE
# todo set hyperparameter for each feature type
for feat in bpm ecg resp biosignals 
do
    python3 main.py --task personalisation --feature $feat \
                    --normalize --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu
    python3 main.py --task personalisation --feature $feat \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu
done
