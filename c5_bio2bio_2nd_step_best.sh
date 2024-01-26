#!/bin/bash
#SBATCH --time=05:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c5_2nd_step_%A.out
#SBATCH --job-name=muse_c5_2
#SBATCH -n 1

module load miniconda

source activate muse

#### --model_id and --checkpoint_seed must be adapted accordingly

#### BPM_normalized
python3 personalisation.py --model_id RNN_2024-01-04-14-16_[ECG]_[BPM_normalized]]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 103 --emo_dim BPM_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

python3 personalisation.py --model_id RNN_2024-01-04-14-28_[resp]_[BPM_normalized]]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 102 --emo_dim BPM_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

#### ECG_normalized
python3 personalisation.py --model_id RNN_2024-01-04-14-09_[BPM]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 103 --emo_dim ECG_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu 

python3 personalisation.py --model_id RNN_2024-01-04-14-31_[resp]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 105 --emo_dim ECG_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu 

#### resp_normalized

python3 personalisation.py --model_id RNN_2024-01-04-14-12_[BPM]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 105 --emo_dim resp_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu                     

python3 personalisation.py --model_id RNN_2024-01-04-14-25_[ECG]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 105 --emo_dim resp_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu   