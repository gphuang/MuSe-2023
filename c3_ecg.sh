#!/bin/bash
#SBATCH --time=00:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env

### AROUSAL

## ECG
python3 main.py --task personalisation \
        --eval_model RNN_2024-01-03-12-42_[ECG]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
        --feature ECG --normalize --predict --eval_seed 104 --use_gpu  

python3 personalisation.py --model_id RNN_2024-01-03-12-42_[ECG]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 104 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 50 --hop_len 25 \
                --use_gpu

python3 personalisation.py --model_id RNN_2024-01-03-12-42_[ECG]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
        --eval_personalised 104_personalised_x \
        --emo_dim physio-arousal

## ECG-mfcc-egemaps
# arousal 0.4821 win_len 50  hop_len 25
python3 main.py --task personalisation \
        --eval_model RNN_2024-04-22-09-45_[egemaps-ecg]_[physio-arousal]_[256_3_True_64]_[0.005_256] \
        --feature egemaps-ecg --normalize --predict --eval_seed 103 --use_gpu

python3 personalisation.py --model_id RNN_2024-04-22-09-45_[egemaps-ecg]_[physio-arousal]_[256_3_True_64]_[0.005_256] \
        --normalize --checkpoint_seed 103 \
        --emo_dim physio-arousal --lr 0.002 \
        --early_stopping_patience 10 \
        --epochs 100 --win_len 50 --hop_len 25 \
        --use_gpu

python3 personalisation.py --model_id RNN_2024-04-22-09-45_[egemaps-ecg]_[physio-arousal]_[256_3_True_64]_[0.005_256] \
        --eval_personalised 103_personalised_2024-04-23-10-45-51 \
        --emo_dim physio-arousal

# valence 0.4388 win_len 200 hop_len 10
python3 main.py --task personalisation \
        --eval_model RNN_2024-04-22-16-14_[mfcc-ecg]_[valence]_[256_3_True_64]_[0.005_256]\
        --feature mfcc-ecg --normalize --predict --eval_seed 101 --use_gpu

python3 personalisation.py --model_id RNN_2024-04-22-16-14_[mfcc-ecg]_[valence]_[256_3_True_64]_[0.005_256]\
        --normalize --checkpoint_seed 101 \
        --emo_dim physio-arousal --lr 0.002 \
        --early_stopping_patience 10 \
        --epochs 100 --win_len 200 --hop_len 10 \
        --use_gpu

python3 personalisation.py --model_id RNN_2024-04-22-09-45_[egemaps-ecg]_[physio-arousal]_[256_3_True_64]_[0.005_256] \
        --eval_personalised 101_personalised_x \
        --emo_dim physio-arousal
