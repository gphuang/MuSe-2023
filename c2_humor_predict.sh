#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=ser
#SBATCH -n 1

module load miniconda

source activate muse

# video
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-41_[faus]_[32_4_True_64]_[0.005_256] \
                --feature faus --eval_seed 103 --predict
        
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-43_[vit]_[64_2_False_64]_[0.0001_256] \
                --feature vit --normalize --eval_seed 105 --predict
        
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-49_[facenet]_[64_4_False_64]_[0.005_256] \
                --feature facenet --eval_seed 105 --predict

# audio
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-23_[egemaps]_[32_2_False_64]_[0.005_256] \
                --feature egemaps --normalize --eval_seed 105 --predict #  --cache

python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-26_[ds]_[256_1_False_64]_[0.001_256] \
                --feature ds --eval_seed 105 --predict

python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-30_[w2v-msp]_[128_2_False_64]_[0.005_256] \
                --feature w2v-msp --eval_seed 105 --predict

# text
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-35_[bert-multilingual]_[128_4_False_64]_[0.001_256] \
                --feature bert-multilingual --eval_seed 105 --predict