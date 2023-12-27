#!/bin/bash
#SBATCH --time=20:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c2_%A.out
#SBATCH --job-name=muse_c2
#SBATCH -n 1

source activate data2vec

python3 main.py --task humor --feature egemaps --normalize --model_dim 32 --rnn_n_layers 2 --lr 0.005 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0.5 --use_gpu

python3 main.py --task humor --feature ds --model_dim 256 --rnn_n_layers 1 --lr 0.001 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0 --use_gpu

python3 main.py --task humor --feature w2v-msp --model_dim 128 --rnn_n_layers 2 --lr 0.005 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0 --use_gpu

python3 main.py --task humor --feature bert-multilingual --model_dim 128 --rnn_n_layers 4 --lr 0.001 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0 --use_gpu

python3 main.py --task humor --feature faus --model_dim 32 --rnn_n_layers 4 --rnn_bi --lr 0.005 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0.5 --use_gpu

python3 main.py --task humor --feature vit --normalize --model_dim 64 --rnn_n_layers 2 --lr 0.0001 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0 --use_gpu

python3 main.py --task humor --feature facenet --model_dim 64 --rnn_n_layers 4 --lr 0.005 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0.5 --use_gpu