#!/bin/bash
#SBATCH --time=05:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env

            
python data_preprocesser.py --feat_extractor egemaps

python3 main.py --task personalisation --feature egemaps_ecg --normalize \
            --emo_dim valence --model_dim 128 --rnn_n_layers 2 --rnn_bi --lr 0.005 \
            --model_type RNN --win_len 100 --hop_len 25 --rnn_dropout 0.5  --use_gpu

"""
for feat in resp mfcc_resp egemaps_resp
do
echo $feat
/usr/bin/time -v python3 main.py --task personalisation --feature $feat --normalize \
            --emo_dim valence --model_dim 256 --rnn_n_layers 2 --rnn_bi --lr 0.005 \
            --model_type RNN --win_len 100 --hop_len 25 --rnn_dropout 0.5 --use_gpu
done"""

# ecg-valence
## ecg 0.1029
## mfcc_ecg 30923409 0.3183
## egemaps_ecg 30947936 

# resp-valence 30939864
## resp 0.0537 (old)
## mfcc_resp
## egemaps_ecg 

## rnn vs. crnn add conv layer between linear and rnn layers, no pooling

# https://arxiv.org/abs/2103.02183

                
