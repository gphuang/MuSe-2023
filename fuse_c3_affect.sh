#!/bin/bash
#SBATCH --time=05:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3_eval_%A.out
#SBATCH --job-name=muse_c3
#SBATCH -n 1

module load miniconda

source activate muse

### AROUSAL

# egemaps, ds run 2nd step, output folder structure?
python3 late_fusion.py --task personalisation --emo_dim physio-arousal \
        --model_ids RNN_2023-12-21-09-06_[egemaps]_[physio-arousal]_[256_4_False_64]_[0.002_256] RNN_2023-12-21-09-37_[faus]_[physio-arousal]_[128_4_True_64]_[0.005_256] \
        --personalised 101_personalised_2024-01-23-12-19-21 103_personalised_2023-12-25-20-12-48
        

