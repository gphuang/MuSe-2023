#!/bin/bash
#SBATCH --time=10:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3_1st_step_%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda

source activate muse 

# rerun missing resp

python3 personalisation.py --model_id RNN_2024-01-24-15-28_[resp]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 103 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 10 --hop_len 5 \
                --use_gpu
                
python3 personalisation.py --model_id RNN_2024-01-24-15-31_[resp]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 103 \
                --emo_dim valence --lr 0.001 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

                