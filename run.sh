#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/hyper_%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda

source activate muse

python3 main.py --task personalisation --feature BPM \
                    --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.01 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu
                