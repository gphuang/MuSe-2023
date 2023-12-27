#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3_bio_2nd_%A.out
#SBATCH --job-name=muse_c3_bio
#SBATCH -n 1

source activate data2vec

### biosignals for AROUSAL & VALENCE
# todo set hyperparameter for each feature type

#### --model_id and --checkpoint_seed must be adapted accordingly
python3 personalisation.py --model_id RNN_2023-12-21-14-55_[bpm]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 101 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 10 --hop_len 5 \
                --use_gpu
python3 personalisation.py --model_id RNN_2023-12-21-14-55_[bpm]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 102 \
                --emo_dim valence --lr 0.001 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

python3 personalisation.py --model_id RNN_2023-12-21-14-55_[ecg]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 101 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 10 --hop_len 5 \
                --use_gpu
python3 personalisation.py --model_id RNN_2023-12-21-14-55_[ecg]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 102 \
                --emo_dim valence --lr 0.001 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

python3 personalisation.py --model_id RNN_2023-12-21-14-55_[resp]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 101 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 10 --hop_len 5 \
                --use_gpu
python3 personalisation.py --model_id RNN_2023-12-21-14-59_[resp]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 102 \
                --emo_dim valence --lr 0.001 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

python3 personalisation.py --model_id RNN_2023-12-21-15-02_[biosignals]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 101 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 10 --hop_len 5 \
                --use_gpu
python3 personalisation.py --model_id RNN_2023-12-21-15-05_[biosignals]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 102 \
                --emo_dim valence --lr 0.001 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu