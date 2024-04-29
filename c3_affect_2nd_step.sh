#!/bin/bash
#SBATCH --time=48:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env

#### --model_id and --checkpoint_seed must be adapted accordingly

### AROUSAL

# egemaps
python3 personalisation.py --model_id RNN_2023-12-21-09-06_[egemaps]_[physio-arousal]_[256_4_False_64]_[0.002_256] --normalize \
                        --checkpoint_seed 101 --emo_dim physio-arousal \
                        --lr 0.002 --early_stopping_patience 10 --epochs 100 \
                        --win_len 10 --hop_len 5 --use_gpu

# ds
python3 personalisation.py --model_id RNN_2023-12-21-09-11_[ds]_[physio-arousal]_[32_2_False_64]_[0.005_256] \
                        --checkpoint_seed 102 --emo_dim physio-arousal \
                        --lr 0.005 --early_stopping_patience 10 --epochs 100 \
                        --win_len 20 --hop_len 10 --use_gpu

# w2v
python3 personalisation.py --model_id RNN_2023-12-21-09-24_[w2v-msp]_[physio-arousal]_[32_4_True_64]_[0.005_256] \
                        --checkpoint_seed 104 --emo_dim physio-arousal \
                        --lr 0.01 --early_stopping_patience 10 --epochs 100 \
                        --win_len 20 --hop_len 10 --use_gpu

# faus
python3 personalisation.py --model_id RNN_2023-12-21-09-37_[faus]_[physio-arousal]_[128_4_True_64]_[0.005_256] \
                        --checkpoint_seed 101 --emo_dim physio-arousal \
                        --lr 0.002 --early_stopping_patience 10 --epochs 100 \
                        --win_len 20 --hop_len 10 --use_gpu

# vit 
python3 personalisation.py --model_id RNN_2023-12-21-09-40_[vit]_[physio-arousal]_[256_4_True_64]_[0.005_256] --normalize \
                        --checkpoint_seed 105 --emo_dim physio-arousal \
                        --lr 0.002 --early_stopping_patience 10 --epochs 100 \
                        --win_len 10 --hop_len 5 --use_gpu

# facenet 
python3 personalisation.py --model_id RNN_2023-12-21-09-46_[facenet]_[physio-arousal]_[256_1_False_64]_[0.001_256] \
                        --checkpoint_seed 101 --emo_dim physio-arousal \
                        --lr 0.001 --early_stopping_patience 10 --epochs 100 \
                        --win_len 10 --hop_len 5 --use_gpu

# BPM
python3 personalisation.py --model_id RNN_2024-04-24-06-59_[melspec-bpm]_[physio-arousal]_[512_4_True_64]_[0.001_256] \
                --normalize --checkpoint_seed 101 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu
# ECG
python3 personalisation.py --model_id RNN_2024-04-24-01-25_[melspec-ecg]_[physio-arousal]_[128_2_True_64]_[0.005_256] \
                --normalize --checkpoint_seed 103 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

# resp
python3 personalisation.py --model_id RNN_2024-04-23-09-04_[egemaps-resp]_[physio-arousal]_[512_4_True_64]_[0.005_256] \
                --normalize --checkpoint_seed 105 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

# hubert
python3 personalisation.py --model_id RNN_2024-04-25-18-16_[hubert-wav]_[physio-arousal]_[512_2_True_64]_[0.005_256] \
                --normalize --checkpoint_seed 104 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

### VALENCE 

# egemaps
python3 personalisation.py --model_id RNN_2023-12-21-09-50_[egemaps]_[valence]_[256_4_False_64]_[0.002_256] --normalize \
                        --checkpoint_seed 101 --emo_dim valence \
                        --lr 0.001 --early_stopping_patience 10 --epochs 100 \
                        --win_len 20 --hop_len 10 --use_gpu

# ds
python3 personalisation.py --model_id RNN_2023-12-21-09-55_[ds]_[valence]_[64_2_False_64]_[0.001_256] \
                        --checkpoint_seed 102 --emo_dim valence \
                        --lr 0.001 --early_stopping_patience 10 --epochs 100 \
                        --win_len 20 --hop_len 10 --use_gpu

# w2v
python3 personalisation.py --model_id RNN_2023-12-21-10-01_[w2v-msp]_[valence]_[128_4_True_64]_[0.005_256] \
                        --checkpoint_seed 104 --emo_dim valence \
                        --lr 0.002 --early_stopping_patience 10 --epochs 100 \
                        --win_len 20 --hop_len 10 --use_gpu

# faus
python3 personalisation.py --model_id RNN_2023-12-21-10-08_[faus]_[valence]_[128_4_True_64]_[0.005_256] \
                        --checkpoint_seed 101 --emo_dim valence \
                        --lr 0.001 --early_stopping_patience 10 --epochs 100 \
                        --win_len 20 --hop_len 10 --use_gpu

# vit 
python3 personalisation.py --model_id RNN_2023-12-21-10-13_[vit]_[valence]_[128_4_True_64]_[0.001_256] --normalize \
                        --checkpoint_seed 103 --emo_dim valence \
                        --lr 0.001 --early_stopping_patience 10 --epochs 100 \
                        --win_len 10 --hop_len 5 --use_gpu


# facenet 
python3 personalisation.py --model_id RNN_2023-12-21-10-19_[facenet]_[valence]_[128_2_False_64]_[0.005_256] \
                        --checkpoint_seed 102 --emo_dim valence \
                        --lr 0.002 --early_stopping_patience 10 --epochs 100 \
                        --win_len 10 --hop_len 5 --use_gpu

# BPM 
python3 personalisation.py --model_id RNN_2024-04-27-00-41_[melspec-bpm]_[valence]_[128_4_True_64]_[0.005_256].txt \
                --normalize --checkpoint_seed 104 \
                --emo_dim valence --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

# ECG
python3 personalisation.py --model_id RNN_2024-04-24-02-02_[mfcc-ecg]_[valence]_[256_4_True_64]_[0.001_256] \
                --normalize --checkpoint_seed 105 \
                --emo_dim valence --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

# resp
python3 personalisation.py --model_id RNN_2024-01-24-15-31_[resp]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 103 \
                --emo_dim valence --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

# hubert
python3 personalisation.py --model_id RNN_2024-04-26-00-37_[hubert-wav]_[valence]_[512_2_True_64]_[0.001_256] \
                --normalize --checkpoint_seed 102 \
                --emo_dim valence --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu                        
