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

# add cnn layer

python3 main.py --task personalisation --feature BPM \
                    --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.001 --win_len 25 --hop_len 10 \
                    --rnn_dropout 0.5 \
                    --use_gpu

                
