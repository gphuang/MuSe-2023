#!/bin/bash
#SBATCH --time=05:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c4_av_to_bio_2nd_%A.out
#SBATCH --job-name=muse_c4
#SBATCH -n 1

module load miniconda

source activate muse

#### BPM_normalized 

# egemaps
python3 personalisation.py --model_id RNN_2023-12-21-14-55_[egemaps]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 105 --emo_dim BPM_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# ds
python3 personalisation.py --model_id RNN_2023-12-21-15-02_[ds]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 104 --emo_dim BPM_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# w2v-msp
python3 personalisation.py --model_id RNN_2023-12-21-15-10_[w2v-msp]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 105 --emo_dim BPM_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# faus
python3 personalisation.py --model_id RNN_2023-12-21-15-20_[faus]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 103 --emo_dim BPM_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# vit
python3 personalisation.py --model_id RNN_2023-12-21-15-27_[vit]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 105 --emo_dim BPM_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# facenet
python3 personalisation.py --model_id RNN_2023-12-21-15-34_[facenet]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 104 --emo_dim BPM_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu
 
#### ECG_normalized 

# egemaps
python3 personalisation.py --model_id RNN_2023-12-21-14-57_[egemaps]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 105 --emo_dim ECG_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# ds
python3 personalisation.py --model_id RNN_2023-12-21-15-06_[ds]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 101 --emo_dim ECG_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# w2v-msp
python3 personalisation.py --model_id RNN_2023-12-21-15-14_[w2v-msp]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 101 --emo_dim ECG_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# faus
python3 personalisation.py --model_id RNN_2023-12-21-15-23_[faus]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 104 --emo_dim ECG_normalized \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# vit
python3 personalisation.py --model_id RNN_2023-12-21-15-30_[vit]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 101 --emo_dim ECG_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# facenet
python3 personalisation.py --model_id RNN_2023-12-21-15-38_[facenet]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 105 --emo_dim ECG_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu 

#### resp_normalized

# egemaps
python3 personalisation.py --model_id RNN_2023-12-22-15-57_[egemaps]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 102 --emo_dim resp_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu

# ds
python3 personalisation.py --model_id RNN_2023-12-22-16-01_[ds]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 101 --emo_dim resp_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu                            

# w2v-msp
python3 personalisation.py --model_id RNN_2023-12-22-16-08_[w2v-msp]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 103 --emo_dim resp_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu                             

# faus
python3 personalisation.py --model_id RNN_2023-12-22-16-13_[faus]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 101 --emo_dim resp_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu                             

# vit
python3 personalisation.py --model_id RNN_2023-12-22-16-17_[vit]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 105 --emo_dim resp_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu                             

# facenet
python3 personalisation.py --model_id RNN_2023-12-22-16-22_[facenet]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                            --normalize --checkpoint_seed 102 --emo_dim resp_normalized  \
                            --lr 0.002 --early_stopping_patience 10 \
                            --epochs 100 --win_len 10 --hop_len 5 \
                            --use_gpu                                                                                                                                 
