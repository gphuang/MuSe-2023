#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c1_facenet_%A.out
#SBATCH --job-name=facenet
#SBATCH -n 1

source activate data2vec

python3 main.py --task mimic --feature facenet  \
                --model_dim 32 --rnn_n_layers 1 --lr 0.005 --rnn_dropout 0.  \
                --early_stopping_patience 10 --reduce_lr_patience 5  \
                --use_gpu
