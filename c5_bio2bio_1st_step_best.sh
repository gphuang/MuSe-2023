#!/bin/bash
#SBATCH --time=10:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c5_1st_step_%A.out
#SBATCH --job-name=muse_c5_1
#SBATCH -n 1

source activate data2vec

### biosignals mapping
# todo set hyperparameter for each feature type

for feat in BPM ECG resp 
do
for biosignal in BPM_normalized ECG_normalized resp_normalized
do
    python3 main.py --task personalisation --feature $feat \
                    --normalize --emo_dim $biosignal \
                    --model_dim 256 --rnn_n_layers 4 \
                    --lr 0.002  --win_len 50 \
                    --hop_len 25 --rnn_dropout 0.5\
                    --use_gpu
done 
done