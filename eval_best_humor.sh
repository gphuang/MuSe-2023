#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3_eval_%A.out
#SBATCH --job-name=muse_c3
#SBATCH -n 1

source activate data2vec

# best audio
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-30_[w2v-msp]_[128_2_False_64]_[0.005_256] \
                --feature w2v-msp --eval_seed 105 --predict

# best video
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-43_[vit]_[64_2_False_64]_[0.0001_256] \
                --feature vit --normalize --eval_seed 105 --predict

# best text
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-35_[bert-multilingual]_[128_4_False_64]_[0.001_256] \
                --feature bert-multilingual --eval_seed 105 --predict