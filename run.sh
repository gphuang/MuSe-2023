#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda

source activate muse

python3 main.py --task mimic --feature egemaps  --normalize --model_type AttnRNN \
                --model_dim 256 --rnn_n_layers 2 --lr 0.001 --rnn_dropout 0.5  \
                --early_stopping_patience 10 --reduce_lr_patience 5 \
                --use_gpu 
                