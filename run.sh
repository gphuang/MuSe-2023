#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda

source activate muse 

# add cnn layer

python3 main.py --task personalisation --feature BPM \
                    --emo_dim physio-arousal \
                    --model_type CRNN \
                    --model_dim 256 --rnn_n_layers 2 \
                    --lr 0.002 --win_len 25 --hop_len 5 \
                    --rnn_dropout 0.5 \
                    --use_gpu

                
