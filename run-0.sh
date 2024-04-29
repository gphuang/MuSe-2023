#!/bin/bash
#SBATCH --time=12:59:59
#SBATCH --mem=125G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env

python3 personalisation.py --model_id RNN_2023-12-21-09-06_[egemaps]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
        --eval_personalised 101_personalised_2024-01-23-12-19-21 \
        --normalize --emo_dim physio-arousal --predict

python3 personalisation.py --model_id RNN_2023-12-21-09-37_[faus]_[physio-arousal]_[128_4_True_64]_[0.005_256] \
        --eval_personalised 103_personalised_2023-12-25-20-12-48 \
        --normalize --emo_dim physio-arousal --predict

python3 late_fusion.py --task personalisation --emo_dim physio-arousal \
        --model_ids RNN_2023-12-21-09-06_[egemaps]_[physio-arousal]_[256_4_False_64]_[0.002_256] RNN_2023-12-21-09-37_[faus]_[physio-arousal]_[128_4_True_64]_[0.005_256] \
        --personalised 101_personalised_2024-01-23-12-19-21 103_personalised_2023-12-25-20-12-48

# 29-Apr-2024 ValueError: operands could not be broadcast together with shapes (480,) (240,)

