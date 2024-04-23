#!/bin/bash
#SBATCH --time=10:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3_2nd_step_%A.out
#SBATCH --job-name=muse_c3_2
#SBATCH -n 1

source activate data2vec

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
python3 personalisation.py --model_id RNN_2024-01-03-12-36_[BPM]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 105 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 10 --hop_len 5 \
                --use_gpu
# ECG
python3 personalisation.py --model_id RNN_2024-01-03-12-42_[ECG]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 104 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 50 --hop_len 25 \
                --use_gpu

# resp
python3 personalisation.py --model_id RNN_2024-01-24-15-28_[resp]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 103 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 10 --hop_len 5 \
                --use_gpu

# biosignals
python3 personalisation.py --model_id RNN_2023-12-21-15-02_[biosignals]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 104 \
                --emo_dim physio-arousal --lr 0.002 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 10 --hop_len 5 \
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
python3 personalisation.py --model_id RNN_2024-01-03-12-39_[BPM]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 103 \
                --emo_dim valence --lr 0.001 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

# ECG
python3 personalisation.py --model_id RNN_2024-01-03-12-45_[ECG]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 105 \
                --emo_dim valence --lr 0.001 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

# resp
python3 personalisation.py --model_id RNN_2024-01-24-15-31_[resp]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 103 \
                --emo_dim valence --lr 0.001 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu

# biosignals
python3 personalisation.py --model_id RNN_2023-12-21-15-05_[biosignals]_[valence]_[256_4_False_64]_[0.002_256] \
                --normalize --checkpoint_seed 103 \
                --emo_dim valence --lr 0.001 \
                --early_stopping_patience 10 \
                --epochs 100 --win_len 20 --hop_len 10 \
                --use_gpu                        
