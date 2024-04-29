#!/bin/bash
#SBATCH --time=07:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env

### AROUSAL

# egemaps
python3 main.py --task personalisation --feature egemaps --normalize \
            --emo_dim physio-arousal --model_dim 256 --rnn_n_layers 4 --lr 0.002 \
            --win_len 50 --hop_len 25 --rnn_dropout 0.5 --use_gpu

# deepspectrum
python3 main.py --task personalisation --feature ds \
            --emo_dim physio-arousal --model_dim 32 --rnn_n_layers 2 --lr 0.005 \
            --win_len 200 --hop_len 100 --rnn_dropout 0. --use_gpu

# w2v
python3 main.py --task personalisation --feature w2v-msp \
            --emo_dim physio-arousal --model_dim 32 --rnn_n_layers 4 --rnn_bi --lr 0.005  \
            --win_len 200 --hop_len 100 --rnn_dropout 0.5 --use_gpu

# faus
python3 main.py --task personalisation --feature faus \
            --emo_dim physio-arousal --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.005  \
            --win_len 200 --hop_len 100 --rnn_dropout 0. --use_gpu

# vit
python3 main.py --task personalisation --feature vit --normalize \
            --emo_dim physio-arousal --model_dim 256 --rnn_n_layers 4 --rnn_bi --lr 0.005  \
            --win_len 50 --hop_len 25 --rnn_dropout 0.5 --use_gpu

# facenet
python3 main.py --task personalisation --feature facenet \
            --emo_dim physio-arousal --model_dim 256 --rnn_n_layers 1 --lr 0.001  \
            --win_len 50 --hop_len 25 --rnn_dropout 0.5 --use_gpu

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

### VALENCE

# egemaps
python3 main.py --task personalisation --feature egemaps --normalize \
            --emo_dim valence --model_dim 256 --rnn_n_layers 4 --lr 0.002  \
            --win_len 200 --hop_len 100 --rnn_dropout 0.5 --use_gpu

# deepspectrum
python3 main.py --task personalisation --feature ds \
            --emo_dim valence --model_dim 64 --rnn_n_layers 2 --lr 0.001  \
            --win_len 100 --hop_len 50 --rnn_dropout 0. --use_gpu

# w2v-msp
python3 main.py --task personalisation --feature w2v-msp \
            --emo_dim valence --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.005  \
            --win_len 100 --hop_len 50 --rnn_dropout 0. --use_gpu

# faus
python3 main.py --task personalisation --feature faus \
            --emo_dim valence --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.005  \
            --win_len 200 --hop_len 100 --rnn_dropout 0. --use_gpu

# vit
python3 main.py --task personalisation --feature vit --normalize \
            --emo_dim valence --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.001  \
            --win_len 200 --hop_len 100 --rnn_dropout 0. --use_gpu

# facenet
python3 main.py --task personalisation --feature facenet \
            --emo_dim valence --model_dim 128 --rnn_n_layers 2 --lr 0.005  \
            --win_len 200 --hop_len 100 --rnn_dropout 0. --use_gpu

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

