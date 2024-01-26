#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c1_%A.out
#SBATCH --job-name=muse_c1
#SBATCH -n 1

module load miniconda

source activate muse

python3 main.py --task mimic --feature egemaps  --normalize \
                --model_dim 256 --rnn_n_layers 2 --lr 0.001 --rnn_dropout 0.5  \
                --early_stopping_patience 10 --reduce_lr_patience 5  \
                --use_gpu 
                
python3 main.py --task mimic --feature deepspectrum  \
                --model_dim 256 --rnn_n_layers 4 --lr 0.0005 --rnn_dropout 0.5  \
                --early_stopping_patience 10 --reduce_lr_patience 5  \
                --use_gpu

python3 main.py --task mimic --feature w2v-msp  \
                --model_dim 128 --rnn_n_layers 2 --lr 0.001 --rnn_dropout 0.5  \
                --early_stopping_patience 10 --reduce_lr_patience 5  \
                --use_gpu

python3 main.py --task mimic --feature faus  \
                --model_dim 256 --rnn_n_layers 4 --lr 0.0005 --rnn_bi --rnn_dropout 0.  \
                --early_stopping_patience 10 --reduce_lr_patience 5  \
                --use_gpu

python3 main.py --task mimic --feature electra  \
                --model_dim 128 --rnn_n_layers 1 --rnn_bi --lr 0.005 --rnn_dropout 0.  \
                --early_stopping_patience 10 --reduce_lr_patience 5  \
                --use_gpu

python3 main.py --task mimic --feature vit --normalize  \
                --model_dim 256 --rnn_n_layers 4 --rnn_bi --lr 0.001 --rnn_dropout 0.5  \
                --early_stopping_patience 10 --reduce_lr_patience 5  \
                --use_gpu

python3 main.py --task mimic --feature facenet  \
                --model_dim 32 --rnn_n_layers 1 --lr 0.005 --rnn_dropout 0.  \
                --early_stopping_patience 10 --reduce_lr_patience 5  \
                --use_gpu
