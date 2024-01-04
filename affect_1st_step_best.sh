#!/bin/bash
#SBATCH --time=20:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3_1st_step_%A.out
#SBATCH --job-name=muse_c3_1
#SBATCH -n 1

source activate data2vec

### biosignals for AROUSAL & VALENCE same as 'personalisation_1st_step_best.sh' with avfeats
# todo set hyperparameter for each feature type

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
python3 main.py --task personalisation --feature BPM \
                    --normalize --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# ECG
python3 main.py --task personalisation --feature ECG \
                    --normalize --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu
# resp
python3 main.py --task personalisation --feature resp \
                    --normalize --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# biosignals
python3 main.py --task personalisation --feature biosignals \
                    --normalize --emo_dim physio-arousal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002 --win_len 50 --hop_len 25 \
                    --rnn_dropout 0.5 \
                    --use_gpu                     

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
python3 main.py --task personalisation --feature BPM \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# ECG                
python3 main.py --task personalisation --feature ECG \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# resp
python3 main.py --task personalisation --feature resp \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu

# biosignals
python3 main.py --task personalisation --feature biosignals \
                    --normalize --emo_dim valence  \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 200 --hop_len 100 \
                    --rnn_dropout 0.5 \
                    --use_gpu

