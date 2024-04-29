#!/bin/bash
#SBATCH --time=11:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env

# BPM
python3 main.py --task personalisation --feature melspec-bpm --normalize \
            --emo_dim physio-arousal --model_dim 512 --rnn_n_layers 4 --lr 0.001 \
            --win_len 200 --hop_len 10 --rnn_bi --rnn_dropout 0.5 --use_gpu

# ECG
python3 main.py --task personalisation --feature melspec-ecg --normalize \
                    --emo_dim physio-arousal --model_dim 128 --rnn_n_layers 2 --lr 0.005 \
                    --win_len 50 --hop_len 10 --rnn_bi --rnn_dropout 0.5 --use_gpu
# resp
python3 main.py --task personalisation --feature egemaps-resp --normalize \
                    --emo_dim physio-arousal --model_dim 512 --rnn_n_layers 4 --lr 0.005 \
                    --win_len 100 --hop_len 25 --rnn_bi --rnn_dropout 0.5 --use_gpu

# hubert
python3 main.py --task personalisation --feature hubert-wav --normalize \
                    --emo_dim physio-arousal --model_dim 512 --rnn_n_layers 2 --lr 0.005 \
                    --win_len 200 --hop_len 25 --rnn_bi --rnn_dropout 0.5 --use_gpu                     


# BPM 
python3 main.py --task personalisation --feature melspec-bpm --normalize \
                    --emo_dim valence --model_dim 128 --rnn_n_layers 4 --lr 0.005  \
                    --win_len 50 --hop_len 10 --rnn_bi --rnn_dropout 0.5 --use_gpu

# ECG           
python3 main.py --task personalisation --feature mfcc-ecg --normalize \
                    --emo_dim valence --model_dim 256 --rnn_n_layers 4 --lr 0.001 \
                    --win_len 100 --hop_len 25 --rnn_bi --rnn_dropout 0.5 --use_gpu

# resp
python3 main.py --task personalisation --feature melspec-resp --normalize \
                    --emo_dim valence --model_dim 256 --rnn_n_layers 4 --lr 0.005 \
                    --win_len 200 --hop_len 10 --rnn_bi --rnn_dropout 0.5 --use_gpu

# hubert
python3 main.py --task personalisation --feature hubert-wav --normalize \
                    --emo_dim valence --model_dim 512 --rnn_n_layers 3 --lr 0.001 \
                    --win_len 100 --hop_len 25 --rnn_bi --rnn_dropout 0.5 --use_gpu


                